import streamlit as st
import pandas as pd
import numpy as np
import h5py
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import time
from PIL import Image
import os

import nltk
import os

# Set NLTK data path
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download required NLTK data
def download_nltk_data():
    try:
        nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
        nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
        nltk.download('punkt_tab', download_dir=nltk_data_path, quiet=True)
        # Verify downloads
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt', download_dir=nltk_data_path)
        nltk.download('stopwords', download_dir=nltk_data_path)
        nltk.download('punkt_tab', download_dir=nltk_data_path)
        nltk.data.path.append(nltk_data_path)
    except Exception as e:
        import traceback
        st.error(f"NLTK data download failed: {str(e)}")
        st.text(traceback.format_exc())
        st.stop()

# Call this function at app startup
download_nltk_data()

# Set page config
st.set_page_config(
    page_title="Truno App Experimental- Journal Recommendation System",
    page_icon="truno-app2.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Preprocessing text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
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
    st.subheader("Recommendation Parameters", help="Customize the parameters to fine-tune the results of your recommendations, after recommendations are defined")
    
    # Number of recommendations slider
    top_n = st.slider(
        "Number of Recommendations",
        min_value=1,
        max_value=50,
        value=15,
        help="Select how many journal recommendations you want to see"
    )
    st.write(" ")
    # Minimum similarity score filter
    min_score = st.slider(
        "Minimum Similarity Score (%)",
        min_value=0,
        max_value=100,
        value=30,
        help="Filter out journals with similarity score below this threshold"
    )
    st.write(" ")
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

    st.divider()
    
    # Additional filters
    st.subheader("Additional Filters", help = "Narrow down your recommendations based on specific criteria, before recommendations are defined")
    
    # Publisher filter with spacing
    publisher_container = st.container(border=True)
    with publisher_container:
        st.text("Journal Publisher", help="Only show journals from specific publisher")
        publisher_filter = st.text_input(
            "Filter by publisher name",
            "",
            label_visibility="collapsed"
        )

    # Spacing between filters
    st.write("")  # Adds vertical space

    # APC Range Filter with improved layout
    apc_container = st.container(border=True)
    with apc_container:
        st.text("APC Budget Range (USD)")
        
        # Initialize session state for APC values
        if 'apc_min' not in st.session_state:
            st.session_state.apc_min = 0
        if 'apc_max' not in st.session_state:
            st.session_state.apc_max = 7000

        # Dual-point range slider
        apc_range = st.slider(
            "Select price range",
            min_value=0,
            max_value=15000,
            value=(st.session_state.apc_min, st.session_state.apc_max),
            step=100,
            help="Drag handles to adjust budget range",
            label_visibility="collapsed"
        )

        # Number inputs in columns
        col1, col2 = st.columns(2)
        with col1:
            new_min = st.number_input(
                "Minimum",
                min_value=0,
                max_value=st.session_state.apc_max,
                value=st.session_state.apc_min,
                step=100,
                help="Minimum APC value",
                key="apc_min_input"
            )
            
        with col2:
            new_max = st.number_input(
                "Maximum",
                min_value=st.session_state.apc_min,
                max_value=15000,
                value=st.session_state.apc_max,
                step=100,
                help="Maximum APC value",
                key="apc_max_input"
            )

        # Synchronization logic
        if apc_range != (st.session_state.apc_min, st.session_state.apc_max):
            st.session_state.apc_min, st.session_state.apc_max = apc_range
            st.rerun()

        if new_min != st.session_state.apc_min or new_max != st.session_state.apc_max:
            st.session_state.apc_min = new_min
            st.session_state.apc_max = new_max
            st.rerun()

        apc_min = st.session_state.apc_min
        apc_max = st.session_state.apc_max

    # Spacing between filters
    st.write("")  # Adds vertical space

    # Quantile filter with improved layout
    quantile_container = st.container(border=True)
    with quantile_container:
        st.text("Journal Quantiles", help="Select journal quantiles to include")
        
        # Create 4 columns for checkboxes
        qcol1, qcol2, qcol3, qcol4 = st.columns(4)
        with qcol1:
            q1 = st.checkbox("Q1", value=True)
        with qcol2:
            q2 = st.checkbox("Q2", value=True)
        with qcol3:
            q3 = st.checkbox("Q3", value=True)
        with qcol4:
            q4 = st.checkbox("Q4", value=True)
    
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
st.title("📚 Journal Recommendation System - TRUNO APP Experimental")
st.warning("This streamlit is experimental for TRUNO App Journal Recommendation users. It is possible that the data used is still old data and has not been updated, even the recommended results can be different if the settings are set. Visit the following url: http://truno-app.my.id:8081/ for the latest data with search history feature")
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
            
            # Load all data first
            embeddings, all_journal_data = load_data()
            
            # Apply filters to the database BEFORE recommendations
            filtered_journal_data = []
            filtered_embeddings = []
            
            for i, journal in enumerate(all_journal_data):
                # 1. Publisher filter
                pub_match = not publisher_filter or publisher_filter.lower() in str(journal.get('publisher', '')).lower()
                
                # 2. APC filter - fixed version
                apc_match = True
                if apc_min > 0 or apc_max < 15000:
                    apc_str = str(journal.get('apc', '0'))
                    if apc_str in ['-', 'N/A', '']:
                        apc_match = apc_min == 0
                    else:
                        try:
                            apc_cleaned = ''.join(c for c in apc_str if c.isdigit() or c == '.')
                            apc_val = float(apc_cleaned) if apc_cleaned else 0
                            apc_match = apc_min <= apc_val <= apc_max
                        except (ValueError, TypeError):
                            apc_match = False
                
                # 3. Quantile filter
                quantile_match = True
                if not (q1 and q2 and q3 and q4):
                    quantile = str(journal.get('indexScopus', '')).upper()
                    quantile_match = (
                        (q1 and 'Q1' in quantile) or
                        (q2 and 'Q2' in quantile) or
                        (q3 and 'Q3' in quantile) or
                        (q4 and 'Q4' in quantile)
                    )
                
                if pub_match and apc_match and quantile_match:
                    filtered_journal_data.append(journal)
                    filtered_embeddings.append(embeddings[i])
            
            if not filtered_journal_data:
                st.warning("No journals match your filter criteria. Try adjusting your filters.")
                st.stop()
            
            # Convert filtered embeddings to numpy array
            filtered_embeddings = np.array(filtered_embeddings)
            
            # Generate recommendations from filtered data
            recommendations = get_recommendations(
                input_abstract, 
                filtered_embeddings, 
                filtered_journal_data, 
                sort_by, 
                top_n,
                min_score
            )
            
            if not recommendations:
                st.warning("No recommendations found after applying all filters.")
            else:
                # Display results
                st.success(f"Found {len(recommendations)} journal recommendations!")
                
                # Metrics row
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_score = np.mean([r['score'] for r in recommendations])
                    st.metric("Average Similarity", f"{avg_score:.1f}%")
                with col2:
                    q1_journals = len([r for r in recommendations if 'Q1' in r.get('indexScopus', '')])
                    st.metric("Q1 Journals", q1_journals)
                with col3:
                    impact_factors = []
                    for r in recommendations:
                        if_val = r.get('impactFactor', '-')
                        if if_val not in ['-', 'N/A']:
                            try:
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
                    "apc": st.column_config.TextColumn("APC (USD)", format="$%s"),
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
                    "citeScore", "impactFactor", "apc","score"
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
                for i, journal in enumerate(recommendations[:top_n]):
                    with st.expander(f"{i+1}. {journal.get('sourceTitle', 'Unknown')}"):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.metric("Similarity Score", f"{journal['score']:.1f}%")
                            st.metric("Cite Score", journal.get('citeScore', 'N/A'))
                            st.metric("Impact Factor", journal.get('impactFactor', 'N/A'))
                            apc_value = journal.get('apc', 'N/A')
                            st.metric("APC", f"${apc_value}" if apc_value not in ['N/A', '-'] else f"{apc_value} Information not found")
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

# Footer information
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
