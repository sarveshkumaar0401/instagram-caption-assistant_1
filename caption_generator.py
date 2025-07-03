import random
import torch
import re
from PIL import Image
from datasets import load_dataset
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    pipeline,
    set_seed
)


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
text_generator = pipeline("text-generation", model="gpt2")
set_seed(42)


ds = load_dataset("kkcosmos/instagram-images-with-captions", split="train")

def limit_caption_length(text, max_words=25):
    words = text.strip().split()
    return " ".join(words[:max_words])

def get_trending_hashtags(context, sample_size=10000, max_tags=10):
    keywords = set(re.findall(r'\w+', context.lower()))
    matching_hashtags = []

    subset = ds.shuffle(seed=random.randint(0, 9999)).select(range(sample_size))

    for item in subset:
        caption = item.get("caption", "").lower()
        if any(word in caption for word in keywords):
            tags = item.get("hashtags", [])
            if isinstance(tags, str):
                tags = tags.split()
            if isinstance(tags, list):
                matching_hashtags.extend(tags)

    if not matching_hashtags:
        fallback = ["#instagood", "#photooftheday", "#love"]
        matching_hashtags = random.sample(fallback, k=min(max_tags, len(fallback)))

    return random.sample(list(set(matching_hashtags)), k=min(len(matching_hashtags), max_tags))


def style_caption(caption):
    styles = [
        f"✨ {caption}, chasing golden vibes.",
        f"📸 {caption}. Moments like these.",
        f"🌿 Just {caption.lower()} things.",
        f"🖤 {caption}. Simply iconic."
    ]
    return random.choice(styles)


def generate_text_caption(prompt):
    styled_prompt = f"Write an Instagram-style caption for: {prompt.strip()}"
    output = text_generator(styled_prompt, max_length=50, num_return_sequences=1, do_sample=True)
    return output[0]['generated_text'].split("for:")[-1].strip()


def generate_captions_and_hashtags(image=None, text=None):
    if image:
        inputs = processor(images=image, return_tensors="pt")
        output = blip_model.generate(**inputs)
        raw_caption = processor.decode(output[0], skip_special_tokens=True)
        limited_caption = limit_caption_length(raw_caption)
        caption = style_caption(limited_caption)
    elif text:
        raw_caption = generate_text_caption(text)
        limited_caption = limit_caption_length(raw_caption)
        caption = style_caption(limited_caption)
    else:
        raise ValueError("No input provided.")

    hashtags = get_trending_hashtags(raw_caption)
    return {
        "Captions": caption,
        "Hashtags": hashtags
    }


