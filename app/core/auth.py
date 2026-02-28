"""
API Key Authentication Module
==============================
Simple API key authentication for sensitive endpoints.
"""

import os
import secrets
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
import logging

logger = logging.getLogger(__name__)

# API Key header scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Get or generate API key
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    # Generate secure random API key if not in env
    API_KEY = secrets.token_urlsafe(32)
    logger.warning("=" * 70)
    logger.warning("⚠️  NO API_KEY found in environment!")
    logger.warning(f"🔑 Generated temporary API key: {API_KEY}")
    logger.warning("💡 Add this to your .env file: API_KEY={key}".format(key=API_KEY))
    logger.warning("=" * 70)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Verify API key from X-API-Key header.

    Raises:
        HTTPException: 401 if API key is missing or invalid

    Returns:
        str: The validated API key
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include X-API-Key header."
        )

    if api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    return api_key
