"""
AI Stylist — Lambda Handler (Gender-Neutral Version 2.0)
=========================================================

Features:
- Amazon Bedrock Claude 3 Sonnet for outfit descriptions
- Amazon Nova Canvas for image generation
- Gender-aware prompting (men/women/neutral)
- CORS support for web frontend
- S3 storage with pre-signed URLs
- Comprehensive error handling

Author: AI Stylist Team
Version: 2.0 (Gender-Neutral)
Updated: February 2026
"""

import json
import boto3
import base64
import uuid
import os
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ═══════════════════════════════════════════════════════════════════════════
# AWS CLIENTS
# ═══════════════════════════════════════════════════════════════════════════

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.environ.get("AWS_REGION", "us-east-1")
)

s3 = boto3.client("s3")

# ═══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT VARIABLES (Set in Lambda Configuration)
# ═══════════════════════════════════════════════════════════════════════════

S3_BUCKET       = os.environ["S3_BUCKET_NAME"]
CLAUDE_MODEL_ID = os.environ.get("CLAUDE_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
NOVA_MODEL_ID   = os.environ.get("NOVA_MODEL_ID", "amazon.nova-canvas-v1:0")
NUM_IMAGES      = int(os.environ.get("NUM_IMAGES", "4"))
IMAGE_EXPIRY    = int(os.environ.get("IMAGE_EXPIRY_SECS", "3600"))


# ═══════════════════════════════════════════════════════════════════════════
# MAIN HANDLER
# ═══════════════════════════════════════════════════════════════════════════

def lambda_handler(event, context):
    """
    Main Lambda entry point for API Gateway requests.
    
    Expected POST body:
    {
        "occasion": "Business Casual",
        "season": "Summer",
        "styles": ["Minimalist", "Classic"],
        "colors": ["Earth Tones"],
        "gender": "neutral",              // NEW: "men", "women", or "neutral"
        "custom_prompt": "Optional text"
    }
    
    Returns:
    {
        "success": true,
        "count": 4,
        "outfits": [
            {
                "name": "Urban Professional",
                "description": "...",
                "image_url": "https://...",
                "gender": "neutral"
            }
        ]
    }
    """
    
    # ───────────────────────────────────────────────────────────────────────
    # CORS Preflight Request (OPTIONS)
    # ───────────────────────────────────────────────────────────────────────
    if event.get("httpMethod") == "OPTIONS":
        return cors_response(200, {})
    
    try:
        # ───────────────────────────────────────────────────────────────────
        # Parse Request Body
        # ───────────────────────────────────────────────────────────────────
        body = json.loads(event.get("body", "{}"))
        logger.info(f"Received request: {json.dumps(body)}")
        
        # Extract user preferences
        occasion = body.get("occasion", "Casual")
        season = body.get("season", "Summer")
        styles = body.get("styles", ["Classic"])
        colors = body.get("colors", ["Neutral"])
        custom_prompt = body.get("custom_prompt", "").strip()
        gender_pref = body.get("gender", "neutral")  # NEW!
        
        # Validate gender preference
        if gender_pref not in ["men", "women", "neutral"]:
            gender_pref = "neutral"
        
        logger.info(f"Preferences: {occasion}, {season}, {styles}, {colors}, Gender: {gender_pref}")
        
        # ───────────────────────────────────────────────────────────────────
        # Generate Outfit Descriptions with Claude
        # ───────────────────────────────────────────────────────────────────
        outfit_descriptions = generate_outfit_descriptions(
            occasion=occasion,
            season=season,
            styles=styles,
            colors=colors,
            custom_prompt=custom_prompt,
            gender=gender_pref
        )
        
        # ───────────────────────────────────────────────────────────────────
        # Generate Images with Nova Canvas
        # ───────────────────────────────────────────────────────────────────
        results = []
        
        for i, outfit in enumerate(outfit_descriptions[:NUM_IMAGES]):
            try:
                image_url = generate_and_upload_image(
                    outfit=outfit,
                    index=i,
                    gender=gender_pref
                )
                
                results.append({
                    "name": outfit.get("name", f"Outfit {i+1}"),
                    "description": outfit.get("description", ""),
                    "image_url": image_url,
                    "gender": gender_pref,
                    "index": i
                })
                
                logger.info(f"✅ Generated outfit {i+1}/{NUM_IMAGES}")
                
            except Exception as img_error:
                logger.error(f"Image generation failed for outfit {i}: {img_error}")
                results.append({
                    "name": outfit.get("name", f"Outfit {i+1}"),
                    "description": outfit.get("description", ""),
                    "image_url": None,
                    "error": str(img_error),
                    "gender": gender_pref,
                    "index": i
                })
        
        # ───────────────────────────────────────────────────────────────────
        # Return Success Response
        # ───────────────────────────────────────────────────────────────────
        return cors_response(200, {
            "success": True,
            "count": len(results),
            "outfits": results,
            "preferences": {
                "occasion": occasion,
                "season": season,
                "styles": styles,
                "colors": colors,
                "gender": gender_pref
            },
            "models_used": {
                "text": CLAUDE_MODEL_ID,
                "image": NOVA_MODEL_ID
            },
            "generated_at": datetime.utcnow().isoformat() + "Z"
        })
    
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in request body: {e}")
        return cors_response(400, {
            "success": False,
            "error": "Invalid JSON in request body",
            "details": str(e)
        })
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return cors_response(500, {
            "success": False,
            "error": str(e),
            "type": type(e).__name__
        })


# ═══════════════════════════════════════════════════════════════════════════
# CLAUDE TEXT GENERATION (Gender-Aware)
# ═══════════════════════════════════════════════════════════════════════════

def generate_outfit_descriptions(occasion, season, styles, colors, custom_prompt, gender):
    """
    Call Claude 3 Sonnet to generate gender-appropriate outfit descriptions.
    
    Args:
        occasion (str): Event type (e.g., "Business Casual")
        season (str): Season/climate (e.g., "Summer")
        styles (list): Style preferences (e.g., ["Minimalist"])
        colors (list): Color palette (e.g., ["Earth Tones"])
        custom_prompt (str): Additional context
        gender (str): "men", "women", or "neutral"
    
    Returns:
        list: List of outfit dictionaries with name, description, image_prompt
    """
    
    # Build gender-specific context
    gender_contexts = {
        "men": "masculine styles, menswear, tailored for male body type, men's fashion",
        "women": "feminine styles, womenswear, tailored for female body type, women's fashion",
        "neutral": "gender-neutral styles, unisex fashion, adaptable silhouettes, inclusive design"
    }
    gender_context = gender_contexts.get(gender, gender_contexts["neutral"])
    
    # Construct system prompt
    system_prompt = f"""You are an expert fashion stylist specializing in {gender_context}.

Generate exactly {NUM_IMAGES} personalized outfit recommendations.

IMPORTANT:
- Tailor ALL recommendations for {gender} styling
- Use clothing items appropriate for {gender} fashion
- Consider body types and fits for {gender}
- Use inclusive, professional language

Return ONLY valid JSON (no markdown, no code fences):

{{
  "outfits": [
    {{
      "name": "Short outfit name (2-4 words)",
      "description": "Professional 2-3 sentence description with specific clothing items, fabrics, colors, and styling details",
      "image_prompt": "Detailed visual description for AI image generation. Must include: {gender_context}, specific clothing items, colors, season-appropriate styling, photography style"
    }}
  ]
}}"""
    
    # Construct user message
    user_message = f"""Create {NUM_IMAGES} outfit recommendations for {gender} fashion based on:

TARGET AUDIENCE: {gender_context}
Occasion: {occasion}
Season: {season}
Style Preferences: {', '.join(styles)}
Color Palette: {', '.join(colors)}
{f'Additional Context: {custom_prompt}' if custom_prompt else ''}

Make each outfit visually distinct and professionally styled for {gender}.
Return only the JSON."""
    
    # Prepare request
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 3000,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_message}
        ]
    }
    
    logger.info(f"Calling Claude for {gender} outfits...")
    
    # Invoke Claude
    response = bedrock.invoke_model(
        modelId=CLAUDE_MODEL_ID,
        body=json.dumps(request_body),
        contentType="application/json",
        accept="application/json"
    )
    
    # Parse response
    response_body = json.loads(response["body"].read())
    raw_text = response_body["content"][0]["text"].strip()
    
    logger.info(f"Claude response preview: {raw_text[:200]}...")
    
    # Clean markdown fences if present
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()
    
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3].strip()
    
    # Parse JSON
    parsed = json.loads(raw_text)
    outfits = parsed.get("outfits", [])
    
    if not outfits:
        raise ValueError("Claude returned no outfits in response")
    
    logger.info(f"✅ Claude generated {len(outfits)} outfit descriptions")
    
    return outfits


# ═══════════════════════════════════════════════════════════════════════════
# NOVA CANVAS IMAGE GENERATION (Gender-Specific)
# ═══════════════════════════════════════════════════════════════════════════

def generate_and_upload_image(outfit, index, gender):
    """
    Generate image with Nova Canvas using gender-appropriate prompts.
    
    Args:
        outfit (dict): Outfit details from Claude
        index (int): Outfit index (0-3)
        gender (str): "men", "women", or "neutral"
    
    Returns:
        str: Pre-signed URL for the generated image
    """
    
    # Get base image prompt from Claude
    image_prompt = outfit.get(
        "image_prompt",
        f"Professional fashion photo, {outfit.get('description', 'stylish outfit')}"
    )
    
    # ───────────────────────────────────────────────────────────────────────
    # Add Gender-Specific Styling (CRITICAL for preventing bias!)
    # ───────────────────────────────────────────────────────────────────────
    
    if gender == "men":
        image_prompt += (
            ". Professional menswear photography. Masculine styling and fit. "
            "Tailored for male body type. Men's fashion presentation. "
            "Studio lighting, clean background."
        )
        negative_prompt = (
            "woman, female, feminine styling, women's clothing, womenswear, "
            "dress, skirt, heels, feminine accessories, "
            "low quality, blurry, distorted, cartoon, anime"
        )
    
    elif gender == "women":
        image_prompt += (
            ". Professional womenswear photography. Feminine styling and fit. "
            "Tailored for female body type. Women's fashion presentation. "
            "Studio lighting, clean background."
        )
        negative_prompt = (
            "man, male, masculine styling, men's clothing, menswear, "
            "overly broad shoulders, masculine accessories, "
            "low quality, blurry, distorted, cartoon, anime"
        )
    
    else:  # neutral
        image_prompt += (
            ". Gender-neutral fashion photography. Unisex styling. "
            "Adaptable silhouettes. Mannequin or flat lay preferred. "
            "No gendered styling cues. Professional presentation."
        )
        negative_prompt = (
            "person, face, human model, man, woman, male, female, "
            "gendered styling, body, "
            "low quality, blurry, distorted, cartoon, anime"
        )
    
    # ───────────────────────────────────────────────────────────────────────
    # Nova Canvas Request
    # ───────────────────────────────────────────────────────────────────────
    
    nova_request = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": image_prompt,
            "negativeText": negative_prompt
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "height": 512,
            "width": 512,
            "cfgScale": 8.5,  # Higher = follows prompt more closely
            "seed": (index + 1) * 42  # Different seed per image
        }
    }
    
    logger.info(f"Calling Nova Canvas for {gender} outfit {index}...")
    logger.info(f"Prompt: {image_prompt[:100]}...")
    
    # Invoke Nova Canvas
    nova_response = bedrock.invoke_model(
        modelId=NOVA_MODEL_ID,
        body=json.dumps(nova_request),
        contentType="application/json",
        accept="application/json"
    )
    
    # Parse response
    response_body = json.loads(nova_response["body"].read())
    image_b64 = response_body["images"][0]
    image_bytes = base64.b64decode(image_b64)
    
    logger.info(f"✅ Nova generated {len(image_bytes)} bytes")
    
    # ───────────────────────────────────────────────────────────────────────
    # Upload to S3
    # ───────────────────────────────────────────────────────────────────────
    
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    s3_key = f"outfits/{gender}/{timestamp}/{unique_id}-outfit-{index}.png"
    
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=image_bytes,
        ContentType="image/png",
        Metadata={
            "outfit_name": outfit.get("name", ""),
            "gender": gender,
            "generated_at": timestamp,
            "model": NOVA_MODEL_ID
        }
    )
    
    logger.info(f"✅ Uploaded to s3://{S3_BUCKET}/{s3_key}")
    
    # ───────────────────────────────────────────────────────────────────────
    # Generate Pre-Signed URL (1 hour expiry)
    # ───────────────────────────────────────────────────────────────────────
    
    presigned_url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": s3_key},
        ExpiresIn=IMAGE_EXPIRY
    )
    
    return presigned_url


# ═══════════════════════════════════════════════════════════════════════════
# CORS RESPONSE HELPER
# ═══════════════════════════════════════════════════════════════════════════

def cors_response(status_code, body):
    """
    Return properly formatted CORS response for API Gateway.
    
    Args:
        status_code (int): HTTP status code
        body (dict): Response body (will be JSON-encoded)
    
    Returns:
        dict: API Gateway response object
    """
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
            "Access-Control-Allow-Methods": "POST,OPTIONS"
        },
        "body": json.dumps(body)
    }
