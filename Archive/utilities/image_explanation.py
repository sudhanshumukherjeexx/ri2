import base64
import os

# encoding image
def EncodeImage(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# analyze image using gpt-4o-mini
def AnalyzeImage(image_path, client, custom_prompt):
    """Analyze image using OpenAI's Vision API with proper error handling"""
    try:
        if not os.path.exists(image_path):
            return "Error: Image file not found"
            
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
        # error handling for API key
        if not client or not client.api_key:
            return "Error: OpenAI API key not configured"
            
        chat_completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": custom_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1024,
        )
        return chat_completion
    except Exception as e:
        return f"Error analyzing image: {str(e)}"


# extract content
def ExtractContent(response):
    """
    Extracts and returns the content from the given response object.

    Parameters:
    - response: The response object received from the API call.

    Returns:
    - A string containing the extracted content, or an appropriate message if content cannot be found.
    """
    if hasattr(response, 'choices') and response.choices:
        first_choice = response.choices[0]
        if hasattr(first_choice, 'message') and first_choice.message:
            content = first_choice.message.content
            return content
        else:
            return "No message found in the first choice."
    else:
        return "No choices found in the response."
