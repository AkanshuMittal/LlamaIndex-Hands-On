import time
import requests
import logging
from typing import Dict, Optional, Any
import config

logger = logging.getLogger(__name__)

def extract_linkedin_profile(
    linkedin_profile_url: Optional[str] = None,
    api_key: Optional[str] = None,
    mock: Optional[bool] =  None
) -> Dict[str, Any]:
    
    """Extract LinkedIn profile data using Proxycurl API or loads premade JSON file.
    
       Args:
           linkedin_profile_url: The LinkedIn profile URL to extract data from.
           api_key: Proxycurl API key(Optional, taken from config if not passed
           mock: Force mock mode (optional, taken from config if not passed).

           Returns:
               Dictionary containig the LinkedIn profile data.
    """

    start_time = time.time()

    try:
        # Decide mock mode from config if not explicitly passed 
        if mock is None:
            mock = config.USE_MOCK_DATA

        if mock:
            logger.info("Using mock data from a premade JSON file")
            mock_url = config.MOCK_DATA_URL

            response = requests.get(mock_url, timeout=30)

        else:
            if not api_key:
                api_key = config.PROXYCURL_API_KEY

            if not api_key:
                raise ValueError(
                    "Proxycurl API key is required when mock mode is False."
                )
            
            if not linkedin_profile_url:
                raise ValueError(
                    "LinkedIn profile URL is required when mock mode is False."
                )
            
            logger.info("Starting to extract the LinkedIn profile...")

            api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"

            headers = {
                "Authorization": f"Bearer {api_key}"
            }

            params = {
                "url": linkedin_profile_url,
                "fallback_to_cache": "on-error",
                "use_cache": "if-present",
                "skills": "include",
                "inferred_salary": "include",
                "personal_email": "include",
                "personal_contact_number": "include"
            }

            logger.info(
                f"Sending API request to Proxycurl at "
                f"{time.time() - start_time:.2f} seconds..."
            )

            response = requests.get(
                api_endpoint,
                headers=headers,
                params=params,
                timeout=10
            )

            logger.info(
                f"Received response at {time.time() - start_time:.2f} seconds..."
            )

            if response.status_code == 200:
                try:
                    data = response.json()

                    data = {
                        k: v
                        for k, v in data.items()
                        if v not in([],"",None)
                        and k not in ["people_also_viewed", "certifications"]
                    }
             
                    if data.get("groups"):
                        for group_dict in data["groups"]:
                            group_dict.pop("profile_pic_url", None)

                    return data 
                
                except ValueError as e:
                    logger.error(f"Error parsing JSON response: {e}")
                    logger.error(
                        f"Response content: {response.text[:200]}..."
                    )

                    return {}
                
            else:
                logger.error(
                    f"Failed to retrieve data."
                    f"Status code: {response.status_code}"
                )
                logger.error(f"Response: {response.text}")
                return {}
            
    except Exception as e:
        logger.error(f"Error in extract_linkedin_profile: {e}")
        return {}
                 

