# Author: Archana Shivashankar
# Contributor(s):
# 
# 
# # Dummy file only for testing purpose to hold security-related functions for webhook signature verification.
# This is separated out for clarity and single-responsibility. 
# Please be aware changes in this file witll affect tests in test_unit.py 
# 

import hmac
import hashlib
import os
import secrets
from dotenv import load_dotenv

# Note: In a real FastAPI app, this secret would be loaded once at startup.
load_dotenv()
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "default_test_secret_12345")

def verify_signature(raw_payload: bytes, signature_header: str) -> bool:
    """
    Verifies the GitHub webhook signature using HMAC SHA-256.
    
    This function uses secrets.compare_digest() for constant-time comparison (Security NFR).
    """
    if not signature_header:
        # Return False if the header is missing
        return False

    # Extract the hex digest part (e.g., 'sha256=' is 7 chars long)
    try:
        method, received_signature = signature_header.split('=', 1)
        if method.lower() != 'sha256':
            return False # Reject if not SHA-256
    except ValueError:
        return False # Invalid header format

    # Calculate the expected signature using the shared secret
    calculated_hmac = hmac.new(
        WEBHOOK_SECRET.encode('utf-8'),
        raw_payload,
        hashlib.sha256
    ).hexdigest()

    # CRITICAL SECURITY STEP: Constant-time comparison to prevent timing attacks
    return secrets.compare_digest(calculated_hmac, received_signature)

def generate_valid_signature(payload: bytes) -> str:
    """
    Helper function used ONLY in testing to generate the correct header value 
    for a given payload and secret.
    """
    calculated_hmac = hmac.new(
        WEBHOOK_SECRET.encode('utf-8'),
        payload,
        hashlib.sha256
    ).hexdigest()
    return f"sha256={calculated_hmac}"
