import json

from groq import Groq

from arxiv_digest.settings import MODEL


def json_completion(client: Groq, prompt: str, schema: dict[str, object]) -> dict[str, object]:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "Return only a JSON object matching this schema. Do not wrap it in markdown.\n"
                + json.dumps(schema),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=800,
        temperature=0.1,
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)