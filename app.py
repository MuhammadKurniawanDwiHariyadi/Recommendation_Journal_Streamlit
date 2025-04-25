import streamlit as st
import pandas as pd
import numpy as np
import h5py
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import re
import time
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Journal Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK data
import nltk
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    import os
    os.makedirs('/home/appuser/nltk_data', exist_ok=True)
    nltk.download('punkt', download_dir='/home/appuser/nltk_data')
    nltk.download('stopwords', download_dir='/home/appuser/nltk_data')
    nltk.data.path.append('/home/appuser/nltk_data')

# Preprocessing text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Load embeddings and journal data
@st.cache_resource
def load_data():
    with h5py.File('journal_recommendation_model3.h5', 'r') as file:
        embeddings = file['embeddings'][:]
        journal_data = {col: file[col][:] for col in file.keys() if col != 'embeddings'}
        for key, value in journal_data.items():
            journal_data[key] = [item.decode('utf-8') for item in value]
        journal_dicts = [{key: journal_data[key][i] for key in journal_data.keys()} 
                        for i in range(len(journal_data['idJournal']))]
    return embeddings, journal_dicts

# Get recommendations
def get_recommendations(abstrak, embeddings, journal_data, sort_by, top_n=5, min_score=None):
    def parse_numeric(value_str, prefer='low'):
        if value_str in ['-', 'N/A', None, '']:
            return float('inf') if prefer == 'low' else float('-inf')
        try:
            cleaned = value_str.replace('.', '').replace(' ', '').replace(',', '.')
            return float(cleaned)
        except:
            return float('inf') if prefer == 'low' else float('-inf')

    model = SentenceTransformer("all-mpnet-base-v2")
    preprocessed_abstrak = preprocess_text(abstrak)
    input_embedding = model.encode(preprocessed_abstrak, convert_to_tensor=True)

    similarity_scores = cos_sim(input_embedding, embeddings)[0]

    recommendations = [
        {
            "score": float(similarity_scores[i]) * 100,  # Multiply by 100 to convert to percentage
            **journal_data[i]
        }
        for i in range(len(journal_data))
    ]

    # Apply minimum score filter if specified
    if min_score is not None:
        recommendations = [r for r in recommendations if r['score'] >= min_score/100]

    # Sort by similarity first
    recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)

    # Apply additional sorting
    if sort_by == 'apc_quantile':
        recommendations = sorted(
            recommendations,
            key=lambda x: (
                x.get('indexScopus', ''),
                parse_numeric(x.get('apc', ''), prefer='low')
            )
        )[:top_n]
    elif sort_by == 'cite_score':
        recommendations = sorted(
            recommendations,
            key=lambda x: parse_numeric(x.get('citeScore', ''), prefer='high'),
            reverse=True
        )[:top_n]
    elif sort_by == 'quantile':
        recommendations = sorted(
            recommendations,
            key=lambda x: x.get('indexScopus', '')
        )[:top_n]
    elif sort_by == 'impact_factor':
        recommendations = sorted(
            recommendations,
            key=lambda x: parse_numeric(x.get('impactFactor', ''), prefer='high'),
            reverse=True
        )[:top_n]
    else:
        recommendations = recommendations[:top_n]

    return recommendations

# Sidebar with additional options
with st.sidebar:
    st.title("⚙️ Settings")
    st.subheader("Recommendation Parameters")
    
    # Number of recommendations slider
    top_n = st.slider(
        "Number of Recommendations",
        min_value=1,
        max_value=50,
        value=15,
        help="Select how many journal recommendations you want to see"
    )
    
    # Minimum similarity score filter
    min_score = st.slider(
        "Minimum Similarity Score (%)",
        min_value=0,
        max_value=100,
        value=30,
        help="Filter out journals with similarity score below this threshold"
    )
    
    # Sorting options
    sort_by = st.selectbox(
        "Sort By:",
        options=[
            ("Similarity", "Similarity"),
            ("Cite Score", "cite_score"),
            ("Impact Factor", "impact_factor"),
            ("Scopus Quantile", "quantile"),
            ("APC and Quantile", "apc_quantile")
        ],
        format_func=lambda x: x[0],
        index=0,
        help="Select how to sort the recommendations"
    )[1]
    
    # Additional filters
    st.subheader("Additional Filters")
    publisher_filter = st.text_input(
        "Filter by Publisher (leave empty for all)",
        "",
        help="Only show journals from specific publisher"
    )
    
    # About section
    st.markdown("---")
    st.subheader("About")
    st.markdown("""
    This journal recommendation system uses state-of-the-art NLP techniques to find the most suitable journals for your research abstract.
    
    **How it works:**
    1. Your abstract is processed and converted to a numerical vector
    2. We compare it with thousands of journal profiles
    3. The system returns the best matches
    """)

# Main content
st.title("📚 Journal Recommendation System")
st.markdown("Find the perfect journal for your research paper based on your abstract.")

# Input section with tabs
tab1, tab2 = st.tabs(["📝 Enter Abstract", "📂 Upload Document"])

with tab1:
    input_abstract = st.text_area(
        "Paste your abstract here:",
        height=200,
        placeholder="Enter your research abstract to find suitable journals..."
    )

with tab2:
    uploaded_file = st.file_uploader(
        "Upload a document (PDF, DOCX, TXT)",
        type=['pdf', 'docx', 'txt'],
        help="Upload your paper and we'll extract the abstract automatically. Make sure your document contains the word abstract or abstrak"
    )
    
    if uploaded_file is not None:
        try:
            extracted_text = ""
            abstract = ""
            
            if uploaded_file.type == "application/pdf":
                # PDF processing with PyPDF2
                try:
                    import PyPDF2
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    extracted_text = "\n".join([page.extract_text() for page in pdf_reader.pages])
                except ImportError:
                    st.error("Please install PyPDF2: pip install PyPDF2")
                    st.stop()  # Changed from return to st.stop()
                    
            elif uploaded_file.type == "text/plain":
                # Plain text processing
                extracted_text = str(uploaded_file.read(), "utf-8")
                
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                # DOCX processing with python-docx
                try:
                    from docx import Document
                    doc = Document(uploaded_file)
                    extracted_text = "\n".join([para.text for para in doc.paragraphs])
                except ImportError:
                    st.error("Please install python-docx: pip install python-docx")
                    st.stop()  # Changed from return to st.stop()
            
            # Abstract detection logic - improved version
            if extracted_text:
                # Create lowercase version for case-insensitive search
                lower_text = extracted_text.lower()
                
                # Find all occurrences of abstract/abstrak
                abstract_positions = []
                for keyword in ["abstract", "abstrak"]:
                    pos = lower_text.find(keyword)
                    if pos != -1:
                        abstract_positions.append(pos)
                
                if abstract_positions:
                    # Get the earliest occurrence
                    abstract_start = min(abstract_positions)
                    
                    # Get the text after "abstract" keyword
                    abstract_candidate = extracted_text[abstract_start:]
                    
                    # Extract the abstract content (stop at next section heading or end of text)
                    # Look for common section endings (introduction, keywords, references)
                    end_markers = ["\n1 ", "\nintroduction", "\nkeywords", "\nreferences", "\nliterature", "\nkata"]
                    end_positions = [abstract_candidate.lower().find(marker) for marker in end_markers]
                    end_positions = [pos for pos in end_positions if pos != -1]
                    
                    if end_positions:
                        abstract_end = min(end_positions)
                        abstract = abstract_candidate[:abstract_end].strip()
                    else:
                        # If no end marker found, take reasonable chunk (about 1-2 paragraphs)
                        paragraphs = [p.strip() for p in abstract_candidate.split('\n') if p.strip()]
                        if len(paragraphs) > 1:
                            abstract = paragraphs[0] + "\n\n" + paragraphs[1]
                        else:
                            abstract = abstract_candidate[:500]
                else:
                    abstract = "Could not automatically detect abstract. Please copy it manually below."
            else:
                abstract = "No text could be extracted from the document."
            
            # Display extracted abstract for editing
            input_abstract = st.text_area(
                "Extracted abstract (edit if needed):",
                value=abstract,
                height=200
            )
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.error("Please ensure the file is not password protected and try again.")

# Recommendation button with progress animation
if st.button("🔍 Find Recommendations", use_container_width=True):
    if input_abstract and input_abstract.strip():
        with st.spinner("Analyzing your abstract and finding the best journal matches..."):
            progress_bar = st.progress(0)
            
            # Simulate progress for better UX
            for percent_complete in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent_complete + 1)
            
            embeddings, journal_data = load_data()
            recommendations = get_recommendations(
                input_abstract, 
                embeddings, 
                journal_data, 
                sort_by, 
                top_n,
                min_score
            )
            
            # Apply publisher filter if specified
            if publisher_filter:
                recommendations = [r for r in recommendations 
                                if publisher_filter.lower() in r.get('publisher', '').lower()]
            
            if not recommendations:
                st.warning("No journals match your criteria. Try adjusting your filters.")
            else:
                # Display results
                st.success(f"Found {len(recommendations)} journal recommendations!")
                
                # Metrics row - FIXED VERSION
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_score = np.mean([r['score'] for r in recommendations])
                    st.metric("Average Similarity", f"{avg_score:.1f}%")
                with col2:
                    q1_journals = len([r for r in recommendations if 'Q1' in r.get('indexScopus', '')])
                    st.metric("Q1 Journals", q1_journals)
                with col3:
                    # Fixed impact factor calculation to handle comma decimals
                    impact_factors = []
                    for r in recommendations:
                        if_val = r.get('impactFactor', '-')
                        if if_val not in ['-', 'N/A']:
                            try:
                                # Replace comma with dot and convert to float
                                num = float(if_val.replace(',', '.'))
                                impact_factors.append(num)
                            except ValueError:
                                continue
                    avg_impact = np.mean(impact_factors) if impact_factors else float('nan')
                    st.metric("Avg Impact Factor", f"{avg_impact:.2f}" if not np.isnan(avg_impact) else "N/A")
                
                # Main results table
                st.subheader("📊 Recommended Journals")
                
                # Enhanced dataframe display
                df_recommendations = pd.DataFrame(recommendations)
                
                # Column configuration
                column_config = {
                    "idJournal": st.column_config.TextColumn("Journal ID"),
                    "sourceTitle": st.column_config.TextColumn("Journal Title", width="large"),
                    "indexScopus": st.column_config.TextColumn("Scopus Index"),
                    "citeScore": st.column_config.NumberColumn("Cite Score", format="%.1f"),
                    "publisher": st.column_config.TextColumn("Publisher"),
                    "apc": st.column_config.TextColumn("APC"),
                    "impactFactor": st.column_config.TextColumn("Impact Factor"),
                    "score": st.column_config.ProgressColumn(
                        "Similarity Score",
                        help="How similar this journal is to your abstract",
                        format="%.3f%%",
                        min_value=0,
                        max_value=100
                    ),
                    "urlJournal": st.column_config.LinkColumn("Journal URL")
                }
                
                # Select which columns to show
                default_columns = [
                    "sourceTitle", "publisher", "indexScopus", 
                    "citeScore", "impactFactor", "score"
                ]
                
                # Let users select which columns they want to see
                with st.expander("⚙️ Select columns to display"):
                    all_columns = list(df_recommendations.columns)
                    visible_columns = st.multiselect(
                        "Choose columns to display:",
                        options=all_columns,
                        default=default_columns
                    )
                
                # Display the interactive dataframe
                st.dataframe(
                    df_recommendations,
                    column_config=column_config,
                    hide_index=True,
                    use_container_width=True,
                    column_order=visible_columns
                )
                
                # Download button
                csv = df_recommendations.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Recommendations as CSV",
                    data=csv,
                    file_name="journal_recommendations.csv",
                    mime="text/csv"
                )
                
                # Journal details expanders
                st.subheader("🔍 Journal Details")
                for i, journal in enumerate(recommendations[:top_n]):  # Show details for top 5
                    with st.expander(f"{i+1}. {journal.get('sourceTitle', 'Unknown')}"):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.metric("Similarity Score", f"{journal['score']:.1f}%")
                            st.metric("Cite Score", journal.get('citeScore', 'N/A'))
                            st.metric("Impact Factor", journal.get('impactFactor', 'N/A'))
                            st.metric("APC", journal.get('apc', 'N/A'))
                        with col2:
                            st.markdown(f"**Publisher:** {journal.get('publisher', 'N/A')}")
                            st.markdown(f"**Scopus Index:** {journal.get('indexScopus', 'N/A')}")
                            st.markdown(f"**Acceptance Rate:** {journal.get('acceptanceRate', 'N/A')}")
                            st.markdown(f"**Focus & Scope:**")
                            st.info(journal.get('focusAndScope', 'Not available'))
                            if journal.get('urlJournal'):
                                st.link_button("Visit Journal Website", journal['urlJournal'])
    else:
        st.warning("Please enter an abstract or upload a document first")

# Add some footer information
st.markdown("---")
st.markdown("""
### About This System
This journal recommendation system uses advanced natural language processing to match your research with suitable academic journals. 
The recommendations are based on semantic similarity between your abstract and journal profiles.

**Key Features:**
- Finds journals that best match your research content
- Filters by impact factor, citation score, and other metrics
- Provides direct links to journal websites
""")
