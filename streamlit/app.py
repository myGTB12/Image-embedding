from qdrant_client import QdrantClient
from io import BytesIO
import streamlit as st
import base64
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

collection_name = "svkara_casts"

# Load model CLIP
@st.cache_resource
def get_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def image_to_vector_clip(image: Image.Image):
    model, processor = get_clip_model()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    vector = outputs[0].numpy()
    return vector / (vector**2).sum()**0.5

if 'selected_record' not in st.session_state:
    st.session_state.selected_record = None

def set_selected_record(new_record):
    st.session_state.selected_record = new_record

@st.cache_resource
def get_client():
    return QdrantClient(
        url=st.secrets.get("qdrant_db_url"),
        api_key=st.secrets.get("qdrant_api_key")
    )

def get_initial_records():
    client = get_client()
    records, _ = client.scroll(
        collection_name=collection_name,
        with_vectors=False,
        limit=12
    )
    return records

def get_similar_records():
    client = get_client()
    if st.session_state.selected_record is not None:
        return client.recommend(
            collection_name=collection_name,
            positive=[st.session_state.selected_record.id],
            limit=12
        )
    return records

def get_bytes_from_base64(base64_string):
    return BytesIO(base64.b64decode(base64_string))

records = get_similar_records() if st.session_state.selected_record is not None else get_initial_records()

st.title("Find similar images using CLIP")

uploaded_file = st.file_uploader("Upload an image to find similar ones", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    vector = image_to_vector_clip(image)

    uploaded_file.seek(0)
    base64_image = base64.b64encode(uploaded_file.read()).decode("utf-8")

    client = get_client()
    search_result = client.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=12
    )

    st.header("Uploaded Image:")
    st.image(image)
    st.divider()

    st.header("Similar Images:")
    column = st.columns(3)
    for idx, record in enumerate(search_result):
        col_idx = idx % 3
        image_bytes = get_bytes_from_base64(record.payload["base64"])
        with column[col_idx]:
            st.image(image_bytes)

column = st.columns(3)
for idx, record in enumerate(records):
    col_idx = idx % 3
    image_bytes = get_bytes_from_base64(record.payload["base64"])
    with column[col_idx]:
        st.image(image_bytes)
        st.button(
            label="Find similar",
            key=record.id,
            on_click=set_selected_record,
            args=[record]
        )