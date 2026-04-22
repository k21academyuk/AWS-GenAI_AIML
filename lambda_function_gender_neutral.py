"""
AI Stylist — Lambda Handler (Gender-Neutral Version 2.0)
=========================================================

Features:
- Amazon Bedrock Claude Sonnet 4.6 for outfit descriptions (us-west-2)
- Stable Diffusion 3.5 Large for image generation (us-west-2)
- Gender-aware prompting (men/women/neutral)
- CORS support for web frontend
- S3 storage with pre-signed URLs
- Content filter handling for Stable Diffusion
- Comprehensive error handling

Author: AI Stylist Team
Version: 2.0 (Gender-Neutral)
Updated: April 2026
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
    region_name="us-west-2"
)

s3 = boto3.client("s3")

# ═══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT VARIABLES (Set in Lambda Configuration)
# ═══════════════════════════════════════════════════════════════════════════

S3_BUCKET          = os.environ["S3_BUCKET_NAME"]
CLAUDE_MODEL_ID    = os.environ.get("CLAUDE_MODEL_ID",    "us.anthropic.claude-sonnet-4.6")
STABILITY_MODEL_ID = os.environ.get("STABILITY_MODEL_ID", "stability.sd3-5-large-v1:0")
NUM_IMAGES         = int(os.environ.get("NUM_IMAGES", "4"))
IMAGE_EXPIRY       = int(os.environ.get("IMAGE_EXPIRY_SECS", "3600"))


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
        "gender": "neutral",
        "custom_prompt": "Optional text"
    }
    """

    if event.get("httpMethod") == "OPTIONS":
        return cors_response(200, {})

    try:
        body = json.loads(event.get("body", "{}"))
        logger.info(f"Received request: {json.dumps(body)}")

        occasion      = body.get("occasion", "Casual")
        season        = body.get("season", "Summer")
        styles        = body.get("styles", ["Classic"])
        colors        = body.get("colors", ["Neutral"])
        custom_prompt = body.get("custom_prompt", "").strip()
        gender_pref   = body.get("gender", "neutral")

        if gender_pref not in ["men", "women", "neutral"]:
            gender_pref = "neutral"

        logger.info(f"Preferences: {occasion}, {season}, {styles}, {colors}, Gender: {gender_pref}")

        # ───────────────────────────────────────────────────────────────────
        # Generate Outfit Descriptions with Claude Sonnet 4.6
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
        # Generate Images with Stable Diffusion 3.5 Large
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
                    "name":        outfit.get("name", f"Outfit {i+1}"),
                    "description": outfit.get("description", ""),
                    "image_url":   image_url,
                    "gender":      gender_pref,
                    "index":       i
                })

                logger.info(f"✅ Generated outfit {i+1}/{NUM_IMAGES}")

            except Exception as img_error:
                logger.error(f"Image generation failed for outfit {i}: {img_error}")
                results.append({
                    "name":        outfit.get("name", f"Outfit {i+1}"),
                    "description": outfit.get("description", ""),
                    "image_url":   None,
                    "error":       str(img_error),
                    "gender":      gender_pref,
                    "index":       i
                })

        return cors_response(200, {
            "success": True,
            "count":   len(results),
            "outfits": results,
            "preferences": {
                "occasion": occasion,
                "season":   season,
                "styles":   styles,
                "colors":   colors,
                "gender":   gender_pref
            },
            "models_used": {
                "text":  CLAUDE_MODEL_ID,
                "image": STABILITY_MODEL_ID
            },
            "generated_at": datetime.utcnow().isoformat() + "Z"
        })

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in request body: {e}")
        return cors_response(400, {
            "success": False,
            "error":   "Invalid JSON in request body",
            "details": str(e)
        })

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return cors_response(500, {
            "success": False,
            "error":   str(e),
            "type":    type(e).__name__
        })


# ═══════════════════════════════════════════════════════════════════════════
# CLAUDE SONNET 4.6 TEXT GENERATION (Gender-Aware)
# ═══════════════════════════════════════════════════════════════════════════

def generate_outfit_descriptions(occasion, season, styles, colors, custom_prompt, gender):

    gender_contexts = {
        "men":     "masculine styles, menswear, tailored for male body type, men's fashion",
        "women":   "feminine styles, womenswear, tailored for female body type, women's fashion",
        "neutral": "gender-neutral styles, unisex fashion, adaptable silhouettes, inclusive design"
    }
    gender_context = gender_contexts.get(gender, gender_contexts["neutral"])

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

    user_message = f"""Create {NUM_IMAGES} outfit recommendations for {gender} fashion based on:

TARGET AUDIENCE: {gender_context}
Occasion: {occasion}
Season: {season}
Style Preferences: {', '.join(styles)}
Color Palette: {', '.join(colors)}
{f'Additional Context: {custom_prompt}' if custom_prompt else ''}

Make each outfit visually distinct and professionally styled for {gender}.
Return only the JSON."""

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 3000,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_message}
        ]
    }

    logger.info(f"Calling Claude Sonnet 4.6 ({CLAUDE_MODEL_ID}) in us-west-2...")

    response = bedrock.invoke_model(
        modelId=CLAUDE_MODEL_ID,
        body=json.dumps(request_body),
        contentType="application/json",
        accept="application/json"
    )

    response_body = json.loads(response["body"].read())
    raw_text      = response_body["content"][0]["text"].strip()

    logger.info(f"Claude Sonnet 4.6 response preview: {raw_text[:200]}...")

    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    if raw_text.endswith("```"):
        raw_text = raw_text[:-3].strip()

    parsed  = json.loads(raw_text)
    outfits = parsed.get("outfits", [])

    if not outfits:
        raise ValueError("Claude Sonnet 4.6 returned no outfits in response")

    logger.info(f"✅ Claude Sonnet 4.6 generated {len(outfits)} outfit descriptions")

    return outfits


# ═══════════════════════════════════════════════════════════════════════════
# STABLE DIFFUSION 3.5 LARGE IMAGE GENERATION (Gender-Specific)
# ═══════════════════════════════════════════════════════════════════════════

def generate_and_upload_image(outfit, index, gender):

    image_prompt = outfit.get(
        "image_prompt",
        f"Professional fashion photo, {outfit.get('description', 'stylish outfit')}"
    )

    if gender == "men":
        image_prompt += (
            ", professional menswear photography, masculine styling and fit, "
            "tailored for male body type, men's fashion presentation, "
            "studio lighting, clean background"
        )
        negative_prompt = (
            "woman, female, feminine styling, women's clothing, womenswear, "
            "dress, skirt, heels, feminine accessories, "
            "low quality, blurry, distorted, cartoon, anime"
        )

    elif gender == "women":
        image_prompt += (
            ", professional womenswear photography, feminine styling and fit, "
            "tailored for female body type, women's fashion presentation, "
            "studio lighting, clean background"
        )
        negative_prompt = (
            "man, male, masculine styling, men's clothing, menswear, "
            "overly broad shoulders, masculine accessories, "
            "low quality, blurry, distorted, cartoon, anime"
        )

    else:  # neutral
        image_prompt += (
            ", gender-neutral fashion photography, unisex styling, "
            "adaptable silhouettes, mannequin or flat lay preferred, "
            "no gendered styling cues, professional presentation"
        )
        negative_prompt = (
            "person, face, human model, man, woman, male, female, "
            "gendered styling, body, "
            "low quality, blurry, distorted, cartoon, anime"
        )

    # ───────────────────────────────────────────────────────────────────────
    # Stable Diffusion 3.5 Large Request
    # ───────────────────────────────────────────────────────────────────────

    sd_request = {
        "prompt":          image_prompt,
        "negative_prompt": negative_prompt,
        "mode":            "text-to-image",
        "aspect_ratio":    "1:1",
        "output_format":   "png",
        "seed":            (index + 1) * 42
    }

    logger.info(f"Calling Stable Diffusion 3.5 Large ({STABILITY_MODEL_ID}) in us-west-2...")
    logger.info(f"Prompt: {image_prompt[:100]}...")

    sd_response = bedrock.invoke_model(
        modelId=STABILITY_MODEL_ID,
        body=json.dumps(sd_request),
        contentType="application/json",
        accept="application/json"
    )

    # ───────────────────────────────────────────────────────────────────────
    # Parse Response — Handle Content Filter
    # ───────────────────────────────────────────────────────────────────────

    response_body  = json.loads(sd_response["body"].read())

    # Check for content filter
    finish_reasons = response_body.get("finish_reasons", [])
    if finish_reasons and finish_reasons[0] in ["Filter reason: prompt", "ERROR"]:
        raise ValueError(f"Image filtered by content moderation: {finish_reasons[0]}")

    if "images" not in response_body or not response_body["images"]:
        raise ValueError("No images returned — possible content filter triggered")

    image_b64   = response_body["images"][0]
    image_bytes = base64.b64decode(image_b64)

    logger.info(f"✅ Stable Diffusion generated {len(image_bytes)} bytes")

    # ───────────────────────────────────────────────────────────────────────
    # Upload to S3
    # ───────────────────────────────────────────────────────────────────────

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    s3_key    = f"outfits/{gender}/{timestamp}/{unique_id}-outfit-{index}.png"

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=image_bytes,
        ContentType="image/png",
        Metadata={
            "outfit_name":  outfit.get("name", ""),
            "gender":       gender,
            "generated_at": timestamp,
            "model":        STABILITY_MODEL_ID
        }
    )

    logger.info(f"✅ Uploaded to s3://{S3_BUCKET}/{s3_key}")

    # ───────────────────────────────────────────────────────────────────────
    # Generate Pre-Signed URL
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
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type":                 "application/json",
            "Access-Control-Allow-Origin":  "*",
            "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
            "Access-Control-Allow-Methods": "POST,OPTIONS"
        },
        "body": json.dumps(body)
    }
