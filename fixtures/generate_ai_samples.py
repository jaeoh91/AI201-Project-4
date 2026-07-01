"""One-off script: generate ~10 genuinely AI-written calibration samples via
Groq and save them to fixtures/ai/*.txt. Run once; not part of the runtime
scoring path (Groq never scores submissions, only explains them later).

Usage: .venv/bin/python fixtures/generate_ai_samples.py
"""

import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

TOPICS = [
    "the benefits of remote work for software teams",
    "how climate change affects coastal cities",
    "why regular exercise improves mental health",
    "the history and future of electric vehicles",
    "how to prepare a healthy weekly meal plan",
    "the importance of cybersecurity for small businesses",
    "why reading fiction builds empathy",
    "how artificial intelligence is changing education",
    "the impact of social media on modern friendships",
    "why urban green spaces matter for community wellbeing",
]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai")


def main():
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    os.makedirs(OUT_DIR, exist_ok=True)

    for i, topic in enumerate(TOPICS, start=1):
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Write a 150-200 word informative paragraph about {topic}. "
                        "Write in a natural essay style."
                    ),
                }
            ],
        )
        text = completion.choices[0].message.content.strip()
        out_path = os.path.join(OUT_DIR, f"ai_{i:02d}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"wrote {out_path} ({len(text.split())} words)")


if __name__ == "__main__":
    main()
