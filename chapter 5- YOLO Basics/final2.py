import os
import cv2
import pytesseract
from ultralytics import YOLO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import spacy
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
IMAGE_PATH = 'images/text6.jpg'  # Path to the input image
YOLO_MODEL_PATH = '../Yolo-Weights/yolov8n.pt'  # Path to YOLO model weights
OCR_CONFIDENCE_THRESHOLD = 80  # Minimum confidence for OCR text recognition
YOLO_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for YOLO object detection
OUTPUT_IMAGE_PATH = 'output/processed_image.jpg'  # Path to save the processed image

# Load NLP Models
nlp = spacy.load("en_core_web_sm")  # For NER and POS tagging
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')  # For semantic similarity and clustering

# --- FUNCTIONS ---
def preprocess_image_for_ocr(image):
    """Convert the image to grayscale, apply thresholding, and Gaussian blur for OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
    return blurred

def extract_text_with_ocr(image, confidence_threshold):
    """Extract text using pytesseract and draw bounding boxes around high-confidence words."""
    config = r'--psm 3'
    data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
    extracted_text = ""
    for i in range(len(data['text'])):
        if int(data['conf'][i]) >= confidence_threshold:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box
            extracted_text += data['text'][i] + " "
    return extracted_text.strip()

def detect_objects_with_yolo(model, image_path, confidence_threshold):
    """Detect objects in an image using YOLO and return a set of detected object names."""
    results = model(image_path, conf=confidence_threshold, show=False)  # Run inference
    detected_objects = set()
    for result in results:
        for box in result.boxes:
            if box.conf[0] >= confidence_threshold:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                detected_objects.add(class_name)
    return detected_objects

def perform_ner(text):
    """Perform Named Entity Recognition (NER) using spaCy."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# def word_frequency(text):
#     """Count word frequencies using CountVectorizer."""
#     vectorizer = CountVectorizer(stop_words="english")
#     word_counts = vectorizer.fit_transform([text]).toarray()[0]
#     words = vectorizer.get_feature_names_out()
#     return Counter(dict(zip(words, word_counts)))
def word_frequency(text):
    """Count word frequencies using CountVectorizer."""
    if not text.strip():  # Check if text is empty or contains only whitespace
        print("Warning: No text provided for word frequency analysis.")
        return Counter()

    vectorizer = CountVectorizer(stop_words="english")
    try:
        word_counts = vectorizer.fit_transform([text]).toarray()[0]
        words = vectorizer.get_feature_names_out()
        return Counter(dict(zip(words, word_counts)))
    except ValueError:
        print("Warning: Vocabulary is empty. The text may contain only stop words.")
        return Counter()


def pos_tagging(text):
    """Perform part-of-speech tagging using spaCy."""
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

# def cluster_words(words, n_clusters=3):
#     """Cluster words based on semantic embeddings."""
#     embeddings = sentence_model.encode(list(words))
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
#     clustered_words = {i: [] for i in range(n_clusters)}
#     for word, label in zip(words, kmeans.labels_):
#         clustered_words[label].append(word)
#     return clustered_words

def cluster_words(words, n_clusters=3):
    """Cluster words based on semantic embeddings."""
    if not words or len(words) < 2:
        print("Warning: Not enough words for clustering. At least two unique words are required.")
        return {0: list(words)}  # Return all words in a single cluster

    embeddings = sentence_model.encode(list(words))

    if len(words) < n_clusters:
        print(f"Warning: Fewer words than clusters. Reducing clusters to {len(words)}.")
        n_clusters = len(words)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)

    clustered_words = {i: [] for i in range(n_clusters)}
    for word, label in zip(words, kmeans.labels_):
        clustered_words[label].append(word)

    return clustered_words


# --- MAIN PROGRAM ---
if __name__ == "__main__":
    # Step 1: Load the image
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found at {IMAGE_PATH}. Please check the path.")
        exit(1)
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Error: Failed to load image at {IMAGE_PATH}.")
        exit(1)
    print("Image loaded successfully.")

    # Step 2: Preprocess the image for OCR
    preprocessed_img = preprocess_image_for_ocr(img)

    # Step 3: Perform OCR and display results
    extracted_text = extract_text_with_ocr(preprocessed_img, OCR_CONFIDENCE_THRESHOLD)
    print("Extracted Text from OCR:")
    print(extracted_text)

    # Step 4: Load the YOLO model
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"Error: YOLO model weights not found at {YOLO_MODEL_PATH}.")
        exit(1)
    model = YOLO(YOLO_MODEL_PATH)
    print("YOLO model loaded successfully.")

    # Step 5: Perform object detection with YOLO
    detected_objects = detect_objects_with_yolo(model, IMAGE_PATH, YOLO_CONFIDENCE_THRESHOLD)
    print("Detected Objects with YOLO:")
    print(", ".join(detected_objects))

    # Step 6: Combine detected objects with extracted text for NLP
    combined_text = extracted_text + " " + " ".join(detected_objects)

    # Validate the combined_text before performing NLP
    if not combined_text.strip():
        print("Error: No valid text extracted from OCR or detected objects.")
    else:
        # 1. Named Entity Recognition
        entities = perform_ner(combined_text)
        print("\nNamed Entities:")
        for entity, label in entities:
            print(f"  {entity} ({label})")

        # 2. Word Frequency Analysis
        word_freq = word_frequency(combined_text)
        if word_freq:
            print("\nWord Frequencies:")
            for word, freq in word_freq.most_common():
                print(f"  {word}: {freq}")

        # 3. Part-of-Speech Tagging
        pos_tags = pos_tagging(combined_text)
        print("\nPart-of-Speech Tags:")
        for word, pos in pos_tags:
            print(f"  {word}: {pos}")

        # 4. Semantic Clustering

        all_words = set(word_freq.keys()).union(detected_objects)
        if all_words:
            clusters = cluster_words(all_words, n_clusters=3)
            print("\nClusters of Words (Semantic Similarity):")
            for cluster_id, cluster_words in clusters.items():
                print(f"  Cluster {cluster_id}: {', '.join(cluster_words)}")
        else:
            print("No words available for clustering.")

    # Step 7: NLP Analysis
    # Named Entity Recognition
    entities = perform_ner(combined_text)
    print("\nNamed Entities:")
    for entity, label in entities:
        print(f"  {entity} ({label})")

    # Word Frequency Analysis
    word_freq = word_frequency(combined_text)
    print("\nWord Frequencies:")
    for word, freq in word_freq.most_common():
        print(f"  {word}: {freq}")

    # Part-of-Speech Tagging
    pos_tags = pos_tagging(combined_text)
    print("\nPart-of-Speech Tags:")
    for word, pos in pos_tags:
        print(f"  {word}: {pos}")

    # Semantic Clustering
    all_words = set(word_freq.keys()).union(detected_objects)
    clusters = cluster_words(all_words, n_clusters=3)
    print("\nClusters of Words (Semantic Similarity):")
    for cluster_id, cluster_words in clusters.items():
        print(f"  Cluster {cluster_id}: {', '.join(cluster_words)}")

    # Step 8: Save and Display the Processed Image
    os.makedirs(os.path.dirname(OUTPUT_IMAGE_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_IMAGE_PATH, img)
    print(f"Processed image saved at {OUTPUT_IMAGE_PATH}.")
    cv2.imshow("Processed Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
