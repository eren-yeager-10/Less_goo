class GeminiService:
    """Service to handle Gemini API calls for translation and classification using the new SDK."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):

        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = MODEL_ID
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        # 1. NEW: Persistent Client (Thread-safe)
        self.client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(
                    attempts=3,          # Use 'attempts' (Not max_attempts)
                    initial_delay=2.0    # Use 'initial_delay' (Not backoff_multiplier)
                )
            )
        )
        
        self.categories_info: CategoriesInfo = {}
        self.all_keywords: KeywordsMap = {}
        self.display_to_key = {}

    async def init_async(self):
        """Initialize categories from DB in async context."""
        try:
            await sync_to_async(self._load_categories, thread_sensitive=True)()
        except Exception as exc:
            logger.exception("Failed to load categories from DB: %s", exc)
            await sync_to_async(self._load_fallback_categories, thread_sensitive=True)()

    def _load_categories(self):
        """Load active categories from database."""
        qs = IssueCategory.objects.filter(is_active=True).select_related("department")
        
        # Initialize local dicts
        categories_info = {}
        all_keywords = {}
        display_to_key_map = {} 

        for cat in qs:
            key = cat.name
            display = cat.display_name
            
            # FIX: Put the lowercase key into our LOCAL map
            display_to_key_map[display.lower().strip()] = key
            
            dept_name = cat.department.name if cat.department else "General"
            keywords = cat.get_keywords_list() if hasattr(cat, "get_keywords_list") else []

            categories_info[key] = {
                "display_name": display,
                "department": dept_name,
                "keywords": keywords,
                "estimated_days": cat.estimated_resolution_days,
            }
            
            for kw in keywords:
                all_keywords.setdefault(kw.lower(), []).append(key)

        # FINAL ASSIGNMENT: Now self.display_to_key will have all your keys
        self.categories_info = categories_info
        self.all_keywords = all_keywords
        self.display_to_key = display_to_key_map
        
        logger.info(f"GeminiService: Indexed {len(categories_info)} categories.")

    def _create_config(self) -> types.GenerateContentConfig:
        # 1. Dynamically build the list of allowed display names from the DB data
        allowed_names = [info['display_name'] for info in self.categories_info.values()]
        
        # 2. Add 'Other Issues' if not in DB to ensure it's always a choice
        if "Other Issues" not in allowed_names:
            allowed_names.append("Other Issues")

        system_instruction = f"""
        You are a civic issue classifier.
        RULES:
        1. Detect language (hi/ta/en) and translate to English.
        2. Choose EXACTLY ONE category from this list: {", ".join(allowed_names)}.
        3. Do not invent new categories. If unsure, use 'Other Issues'.
        4. Assign a priority based on urgency and public safety impact:
            - critical: immediate danger (e.g. live wire, flooding, major accident)
            - high: significant disruption (e.g. main road blocked, water main burst)
            - medium: moderate inconvenience (e.g. pothole, broken streetlight)
            - low: minor or cosmetic issues (e.g. faded road markings, park bench damage)
            Choose EXACTLY ONE from: low, medium, high, critical.
        """

        return types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=GeminiResponse,
            temperature=0.0 # Zero temperature makes the AI follow the list strictly
        )
    
    async def process_issue(self, text: str, location_context: Optional[str] = None) -> GeminiResponse:
        """Process text via Gemini API with deep debug logging."""
        try:
            user_input = f"INPUT: \"{text}\""
            
            # Log the 'Menu' we are sending to Gemini
            allowed_names = [info['display_name'] for info in self.categories_info.values()]
            logger.debug(f"DEBUG: Sending category menu to Gemini: {allowed_names}")

            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=user_input,
                config=self._create_config()
            )
            VALID_PRIORITIES = {"low", "medium", "high", "critical"}
            if response.parsed:
                res = response.parsed
                raw_priority = (res.priority or "medium").lower().strip()
                res.priority = raw_priority if raw_priority in VALID_PRIORITIES else "medium"
                # --- CRITICAL DEBUG LOGS ---
                raw_ai_category = res.category
                normalized_ai_category = raw_ai_category.lower().strip()
                
                logger.info(f"DEBUG: Gemini returned category: '{raw_ai_category}'")
                logger.info(f"DEBUG: Normalized for lookup: '{normalized_ai_category}'")
                logger.info(f"DEBUG: Available keys in map: {list(self.display_to_key.keys())}")
                # ---------------------------

                # Perform the mapping
                mapped_key = self.display_to_key.get(normalized_ai_category)
                
                if mapped_key:
                    logger.info(f"DEBUG: Match found! Mapping to internal key: '{mapped_key}'")
                    res.category = mapped_key
                else:
                    logger.warning(f"DEBUG: No match found for '{normalized_ai_category}'. Defaulting to 'other'.")
                    res.category = 'other'

                if res.confidence_score is None:
                    res.confidence_score = 0.8

                return res
                
            raise ValueError("Empty response from Gemini")

        except Exception as e:
            logger.error(f"DEBUG: Gemini Processing Error: {str(e)}", exc_info=True)
            return await self._fallback_processing(text)
