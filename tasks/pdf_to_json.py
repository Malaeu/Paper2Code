from llm_router.router import wrap_call


def pdf_to_json(path):
    imgs = pdf_to_images(path)
    out = []
    for img in imgs:
        j = wrap_call(
            task="rag",
            prompt={"image": img, "text": "Extract structured JSON"},
            temperature=0,
        )
        out.append(j)
    return merge_pages(out)


def pdf_to_images(path):
    raise NotImplementedError("placeholder")


def merge_pages(pages):
    raise NotImplementedError("placeholder")
