import streamlit as st
import os
import zipfile
import shutil
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
import base64
import google.generativeai as genai
from dotenv import load_dotenv

# Load .env and configure Gemini
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # or use a hardcoded key

# At the top — before any state use
if "cleanup_trigger" not in st.session_state:
    st.session_state["cleanup_trigger"] = False

# Sidebar for custom system prompt
st.sidebar.header("System Prompt")
custom_prompt = st.sidebar.text_area(
    "Override Default",
    value="Extract all readable text from this image. Return only plain text and formatting if necessary.",
    height=200
)


def ocr_with_gemini(image, prompt):
    buf = BytesIO()
    image.save(buf, format="JPEG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content([
        {"inline_data": {"mime_type": "image/jpeg", "data": encoded}},
        prompt
    ])
    
    # Get the text and token count
    text = getattr(response, "text", "⚠️ No OCR output")
    token_count = getattr(response.usage_metadata, "total_token_count", 0)
    
    return text, token_count


def pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img = Image.open(BytesIO(pix.tobytes("png")))
        yield img
    doc.close()

def process_pdfs(input_dir, output_dir, progress_placeholder, status_placeholder):
    total_tokens_used = 0

    structure_output = []
    pdfs = [os.path.join(root, file)
            for root, _, files in os.walk(input_dir)
            for file in files if file.lower().endswith(".pdf")]

    for idx, pdf_file in enumerate(pdfs):
        chapter_name = os.path.splitext(os.path.basename(pdf_file))[0]
        chapter_out = os.path.join(output_dir, chapter_name)
        os.makedirs(chapter_out, exist_ok=True)
        structure_output.append(f"📂 {chapter_name}/")

        images = list(pdf_to_images(pdf_file))
        for i, img in enumerate(images):
            status_placeholder.info(f"🔍 Processing: {chapter_name} (Page {i+1}/{len(images)})")
            text, token_count = ocr_with_gemini(img, prompt=custom_prompt)
            total_tokens_used += token_count

            txt_name = f"{chapter_name}_page_{i+1}.txt"
            txt_path = os.path.join(chapter_out, txt_name)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            structure_output.append(f"   └─ 📄 {txt_name}")

        progress_placeholder.progress((idx + 1) / len(pdfs))

    st.info(f"🔢 Total Gemini Tokens Used: {total_tokens_used}")
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
st.title("PDF Processing Pipeline")

uploaded_zip = st.file_uploader("Upload a ZIP of PDFs", type=["zip"])
if uploaded_zip:
    upload_name = os.path.splitext(uploaded_zip.name)[0]
    extract_dir = os.path.join("output", upload_name)
    zip_path = os.path.join("output", f"{upload_name}.zip")

    # Cleanup if exists
    for path in [extract_dir, zip_path]:
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)

    os.makedirs(extract_dir, exist_ok=True)

    # Save and extract zip
    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.getbuffer())
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    st.success(f"✅ Uploaded and extracted: `{extract_dir}`")

    if st.button("🔍 Start OCR"):
        result_path = os.path.join("output", f"{upload_name}_ocr")
        shutil.rmtree(result_path, ignore_errors=True)
        os.makedirs(result_path)

        progress_placeholder = st.progress(0)
        status_placeholder = st.empty()

        structure = process_pdfs(extract_dir, result_path, progress_placeholder, status_placeholder)
        status_placeholder.success("✅ OCR Complete!")
        st.text_area("📂 Output Structure", structure, height=300)

        # AFTER the text_area and ZIP creation
        zipped = zip_folder(result_path)

        # ➕ Store trigger for download
        if st.download_button("⬇️ Download ZIP", zipped, file_name=f"{upload_name}_ocr.zip", mime="application/zip"):
            # Flag that download was clicked
            st.session_state["cleanup_trigger"] = True
            st.rerun()  # 🔁 Rerun to reach the cleanup block

        # 🔁 Now on next render, cleanup will execute once
        if st.session_state["cleanup_trigger"]:
            try:
                shutil.rmtree(result_path, ignore_errors=True)
                shutil.rmtree(extract_dir, ignore_errors=True)
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                if os.path.exists("output") and not os.listdir("output"):
                    shutil.rmtree("output")
                st.success("🧹 All temporary output files cleaned up.")
            except Exception as e:
                st.warning(f"⚠️ Cleanup failed: {str(e)}")
            finally:
                st.session_state["cleanup_trigger"] = False  # reset after use
