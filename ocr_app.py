import streamlit as st
import os
import zipfile
import shutil
import fitz
from PIL import Image
from io import BytesIO
import base64
import google.generativeai as genai
from dotenv import load_dotenv

# Load .env and configure Gemini
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # or hardcode for now

def ocr_with_gemini(image):
    buf = BytesIO()
    image.save(buf, format="JPEG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content([
        {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": encoded
            }
        },
        "Extract all readable text from this image. Return only plain text without formatting or metadata."
    ])
    return getattr(response, "text", "⚠️ No OCR output")

def pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img = Image.open(BytesIO(pix.tobytes("png")))
        yield img
    doc.close()

def process_pdfs(input_dir, output_dir):
    structure_output = []
    pdfs = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, file))

    progress = st.progress(0)

    for idx, pdf_file in enumerate(pdfs):
        chapter_name = os.path.splitext(os.path.basename(pdf_file))[0]
        chapter_out = os.path.join(output_dir, chapter_name)
        os.makedirs(chapter_out, exist_ok=True)
        structure_output.append(f"📂 {chapter_name}/")

        for i, img in enumerate(pdf_to_images(pdf_file)):
            text = ocr_with_gemini(img)
            txt_name = f"{chapter_name}_page_{i+1}.txt"
            txt_path = os.path.join(chapter_out, txt_name)

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            structure_output.append(f"   └─ 📄 {txt_name}")

        progress.progress((idx + 1) / len(pdfs))

    return "\n".join(structure_output)


def zip_folder(folder_path):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, folder_path)
                zipf.write(abs_path, rel_path)
    zip_buffer.seek(0)
    return zip_buffer

# Streamlit UI
st.title("📖 Gemini OCR PDF Processor")

uploaded_zip = st.file_uploader("Upload a ZIP of PDFs", type=["zip"])
if uploaded_zip:
    upload_name = os.path.splitext(uploaded_zip.name)[0]
    extract_dir = os.path.join("output", upload_name)

    # Cleanup old if exists
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)

    # Save and extract zip locally
    zip_path = os.path.join("output", f"{upload_name}.zip")
    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.getbuffer())
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    st.success(f"✅ Uploaded and extracted to: {extract_dir}")

    if st.button("🔍 Start OCR"):
        result_path = os.path.join("output", f"{upload_name}_ocr")
        if os.path.exists(result_path):
            shutil.rmtree(result_path)
        os.makedirs(result_path)

        structure = process_pdfs(extract_dir, result_path)
        st.text_area("📂 Output Structure", structure, height=300)

        zipped = zip_folder(result_path)
        st.download_button("⬇️ Download ZIP", zipped, file_name=f"{upload_name}_ocr.zip", mime="application/zip")
