import logging
import json
import os
import requests

logger = logging.getLogger(__name__)


# MOCK JSON DATA EXTRACTION

def extract_mock_profile():
    """
    Load LinkedIn profile data from a local mock JSON file.
    Used for development and testing.
    """
    mock_path = "data.json"

    if not os.path.exists(mock_path):
        raise FileNotFoundError(f"Mock data file not found at {mock_path}")

    logger.info("Using mock JSON profile data")

    with open(mock_path, "r", encoding="utf-8") as f:
        return json.load(f)
    

# LINKEDIN PROFILE EXTRACTION 
def extract_linkedin_profile(linkedin_url: str):
    """
    Extract LinkedIn profile data using Apify actor.
    User provides only the LinkedIn URL.
    """
    
    APIFY_API_KEY = os.getenv("APIFY_API_KEY")
    
    if not APIFY_API_KEY:
        raise ValueError("APIFY_API_KEY not found in environment variables")

    logger.info("Extracting LinkedIn profile using Apify")
    
    actor_endpoint = (
        "https://api.apify.com/v2/acts/"
        "VYRyEF4ygTTkaIghe/run-sync-get-dataset-items"
    )

    payload = {
        "startUrls": [{"url": linkedin_url}],
        "resultsLimit": 1
    }
    
    response = requests.post(
        f"{actor_endpoint}?token={APIFY_API_KEY}",
        json=payload,
        timeout=60
    )
    if response.status_code != 200:
        logger.error(f"Apify request failed: {response.text}")
        raise RuntimeError("Failed to extract LinkedIn data via Apify")

    data = response.json()

    if not data:
        raise ValueError("No data returned from Apify")

    # Convert structured data to text for RAG
    return json.dumps(data[0])
               