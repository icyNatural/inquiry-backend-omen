import re

class ScopeProcessor:
    """
    ScopeProcessor applies retrieval bounds to prevent semantic contamination.
    It takes raw retrieved results from ChromaDB or files and filters them
    based on a Scope configuration.
    """
    
    @staticmethod
    def validate_scope_config(scope):
        """
        Ensure scope has all expected fields and sets defaults.
        """
        # Accept either a plain dict or a Pydantic model (Scope)
        if hasattr(scope, "model_dump"):
            scope = scope.model_dump()
        if not isinstance(scope, dict):
            scope = {}
            
        return {
            "namespaces": scope.get("namespaces") or [], # e.g. ["chatgpt_export", "refined_notes"]
            "domains": scope.get("domains") or [],       # e.g. ["career_positioning", "framework"]
            "include_tags": scope.get("include_tags") or [], # e.g. ["#thinkingrefinement"]
            "exclude_tags": scope.get("exclude_tags") or [], # e.g. ["#taoism"]
            "include_terms": scope.get("include_terms") or [], # e.g. ["workflow"]
            "exclude_terms": scope.get("exclude_terms") or []  # e.g. ["Greek Philosophy"]
        }
        
    @classmethod
    def build_chroma_where(cls, scope_config):
        """
        Builds the metadata filter for ChromaDB.
        Chroma metadata fields:
          - source: 'chatgpt_export' | 'refined_notes'
          - domain: 'framework' | 'synthesis' | etc.
        """
        scope = cls.validate_scope_config(scope_config)
        conditions = []
        
        # 1. Filter by namespaces (source metadata in Chroma)
        if scope["namespaces"]:
            if len(scope["namespaces"]) == 1:
                conditions.append({"source": scope["namespaces"][0]})
            else:
                conditions.append({"$or": [{"source": ns} for ns in scope["namespaces"]]})
                
        # 2. Filter by domains (domain metadata in Chroma)
        if scope["domains"]:
            # Standardize domains to include 'general' for items without domain set
            domains_list = list(scope["domains"])
            if len(domains_list) == 1:
                conditions.append({"domain": domains_list[0]})
            else:
                conditions.append({"$or": [{"domain": d} for d in domains_list]})
                
        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}

    @classmethod
    def matches_scope(cls, doc_text, metadata, scope_config):
        """
        Evaluates a single retrieved item against the scope configuration.
        Returns (bool, reason_if_rejected).
        """
        scope = cls.validate_scope_config(scope_config)
        doc_lower = doc_text.lower()
        
        # 1. Namespace Check (redundant if Chroma filter worked, but good safeguard)
        if scope["namespaces"]:
            source = metadata.get("source")
            if source not in scope["namespaces"]:
                return False, f"Namespace '{source}' not in allowed list: {scope['namespaces']}"
                
        # 2. Domain Check
        if scope["domains"]:
            domain = metadata.get("domain") or "general"
            if domain not in scope["domains"]:
                return False, f"Domain '{domain}' not in allowed list: {scope['domains']}"
                
        # 3. Exclude Terms Check (case-insensitive substring match)
        for term in scope["exclude_terms"]:
            term_clean = term.strip().lower()
            if term_clean and term_clean in doc_lower:
                return False, f"Contains excluded term: '{term}'"
                
        # 4. Include Terms Check (must contain all specified terms if any)
        for term in scope["include_terms"]:
            term_clean = term.strip().lower()
            if term_clean and term_clean not in doc_lower:
                return False, f"Missing required term: '{term}'"
                
        # 5. Tag Checks
        note_tags_str = metadata.get("tags") or ""
        note_tags = [t.strip().lower() for t in note_tags_str.split(",") if t.strip()]
        
        # If the item has no metadata tags, check if tags are written inside the document itself (e.g. #tag)
        if not note_tags:
            # simple regex search for #tag
            note_tags = [t.lower() for t in re.findall(r"#\w+", doc_text)]
            
        # Include tags (must match at least one of the specified tags if list is not empty)
        if scope["include_tags"]:
            req_tags = [t.strip().lower() for t in scope["include_tags"]]
            # prepend '#' if missing
            req_tags = [t if t.startswith("#") else f"#{t}" for t in req_tags]
            
            has_matching_tag = any(rt in note_tags for rt in req_tags)
            if not has_matching_tag:
                return False, f"Missing required tags: {scope['include_tags']} (found: {note_tags})"
                
        # Exclude tags (must not contain any of the excluded tags)
        if scope["exclude_tags"]:
            ex_tags = [t.strip().lower() for t in scope["exclude_tags"]]
            ex_tags = [t if t.startswith("#") else f"#{t}" for t in ex_tags]
            
            for et in ex_tags:
                if et in note_tags:
                    return False, f"Contains excluded tag: '{et}'"
                    
        return True, ""

    @classmethod
    def filter_results(cls, results, scope_config):
        """
        Filters list of raw Chroma search results or structured memory chunks.
        Results format: List of dicts, each with:
          - "document": doc string
          - "metadata": metadata dict
          - "distance": optional embedding match score
        """
        filtered = []
        for item in results:
            doc = item.get("document", "")
            meta = item.get("metadata", {})
            
            is_ok, reason = cls.matches_scope(doc, meta, scope_config)
            if is_ok:
                filtered.append(item)
            else:
                # Debug logging can log why items are excluded if needed
                pass
        return filtered
