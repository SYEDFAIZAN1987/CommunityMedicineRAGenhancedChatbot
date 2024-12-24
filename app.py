import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import openai
import streamlit.components.v1 as components


# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Community Medicine Department MMC & RI Chat Bot",
    page_icon="📘",
    layout="wide"
)
# Custom CSS for better styling
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left: 4px solid #002D62;
    }
    .assistant-message {
        background-color: #e8f4f9;
        border-left: 4px solid #c41e3a;
    }
    .source-box {
        background-color: #f8f9fa;
        border: 1px solid #002D62;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin-top: 0.5rem;
        font-size: 0.9em;
    }
    .header-container {
        padding: 1rem;
        background-color: white;
        color: #002D62;
        margin-bottom: 2rem;
        border-bottom: 3px solid #002D62;
    }
    .logo-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
    }
    .logo-item {
        text-align: center;
        flex: 1;
    }
    .logo-item img {
        max-height: 60px;
        object-fit: contain;
    }
    .header-text {
        color: #002D62;
        margin: 0;
        text-align: center;
    }
    .source-tag {
        background-color: #002D62;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8em;
        margin-right: 0.5rem;
    }
    .disclaimer {
        font-size: 0.8em;
        color: #666;
        text-align: center;
        padding: 1rem;
        border-top: 1px solid #eee;
    }
    .divider {
        height: 2px;
        background-color: #002D62;
        margin: 1rem 0;
    }
    .chat-container {
        max-width: 800px;
        margin: 2rem auto;
        padding: 0 1rem;
    }
    .example-button {
        margin: 0.25rem;
    }
    .chat-input {
        max-width: 800px;
        margin: 1rem auto;
    }
    </style>
""", unsafe_allow_html=True)


# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Course Materials", "Project Details"])

# Define the Home Page
if page == "Home":
    # Header Section
    st.markdown("""
        <div class="header-container">
            <div class="divider"></div>
            <h1 class="header-text">ALY 6080 Group 1 Capstone Project</h1>
            <p class="header-text" style="font-size: 1.5em; font-weight: bold; color: red;">
                Built by Syed Faizan, Team Lead of Group 1, ALY 6080, Northeastern University
            </p>
        </div>
    """, unsafe_allow_html=True)

     # Display the logo using st.image
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.write("")  # Empty column for spacing

    with col2:
        st.image("images/univ.png", caption="University Logo", use_container_width=True)

    with col3:
        st.write("")  # Empty column for spacing

    # Display Team Images
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image("images/faizan.jpeg", use_container_width=True, caption="Dr. Syed Faizan")
    with col3:
        st.image("images/DrMudassirAzeezKhan.jpg", use_container_width=True, caption="Dr. Mudassir Azeez Khan")
    

    # Introduction
    st.markdown("""
This knowledge assistant is designed to answer questions related to the Community Medicine Department MMC & RI, covering topics such as:
- Public Health
- Community Health
- Health Equity
- Health Disparities
- Social Determinants of Health

**Note:** To use the RAG-Based Report Query Assistant, the **"ASK"** button must be clicked twice:
1. Once after entering the question in the input box.
2. A second time after the **"Analyzing Sources"** message disappears.

This ensures that the query is fully processed and returns accurate results.
""")


    # Chatbot Interface
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ''

    # Function to handle example question clicks
    def set_example_question(question):
        st.session_state.current_question = question
    
   

# OpenAI API key check
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("⚠️ OpenAI API key not found! Please set the OPENAI_API_KEY environment variable.")
        st.stop()

    openai.api_key = openai_api_key

# Initialize FAISS vector store with dangerous deserialization
    try:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.load_local("db", embeddings, allow_dangerous_deserialization=True)
    # st.success("FAISS vector store loaded successfully!")
    # Commented out the warning to suppress it
    # st.warning(
    #     "⚠️ Dangerous deserialization is enabled. Ensure the FAISS index (`db`) is from a trusted source "
    #     "to avoid potential security risks."
    # )
    except Exception as e:
        st.error(f"⚠️ Error connecting to the database: {str(e)}")
        st.stop()


    def rag(query, n_results=5):
        """RAG function with multi-source context"""
        try:
            docs = vector_store.similarity_search(query, k=n_results)
            joined_information = '; '.join([doc.page_content for doc in docs])
            messages = [
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable assistant with expertise in socio-economic data, housing stability, "
                    "financial trends, and predictive analytics. Provide clear, accurate answers based on provided "
                    "information and your ALY 6080 Group 1 Project context."
                )
            },
            {"role": "user", "content": f"Question: {query}\nInformation: {joined_information}"}
            ]
        
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7
            )
        
            return response.choices[0].message['content'], docs
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")

# Chat interface in container
    chat_container = st.container()

    with chat_container:
    # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"<div class='chat-message user-message'>💭 You: {message['content']}</div>", 
                       unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='chat-message assistant-message'>
                    <div style='margin-bottom: 0.5rem;'>📘 Assistant: {message['response']}</div>
                </div>
            """, unsafe_allow_html=True)

# If there's a current question in the session state, use it as the default value
    user_query = st.text_input(
        label="Ask your question about the ALY 6080 project",
        help="Type your question or click an example below",
        placeholder="Example: What are the key trends in housing stability?",
        value=st.session_state.current_question,
        key="unique_user_input_key"  # Assign a unique key
    )

# Query input with examples
    st.markdown("<div class='chat-input'>", unsafe_allow_html=True)


# Example questions as buttons
    example_questions = [
        "What are the key findings about housing stability?",
        "What are the income trends for Toronto CMA?",
        "How does the living wage prediction look for 2030?",
        "What are the demographic trends observed in GTA?"
    ]

    st.markdown("### 💡 Example Questions")
    cols = st.columns(2)
    for idx, question in enumerate(example_questions):
    # Assign a unique key for each button
        if cols[idx % 2].button(question, key=f"example_question_key_{idx}"):
            st.session_state.current_question = question  # Set the selected example question

# Single "Ask" Button Logic
    if st.button("Ask", type="primary", use_container_width=True):  # Ensure only one button
        if not user_query:
            st.warning("⚠️ Please enter a question!")
        else:
        # Append user's query to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
        
            with st.spinner("🔍 Analyzing sources..."):
                try:
                # Call the RAG function and get response and sources
                    response, sources = rag(user_query)
                
                # Append assistant's response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "response": response,
                        "sources": sources
                    })
                
                # Clear the current question after processing
                    st.session_state.current_question = ''
                
                # Update app state to refresh UI
                    st.session_state['force_rerun'] = True  # Trigger UI update
                except Exception as e:
                # Log and display a detailed error message
                    st.error(f"⚠️ An error occurred while processing your query: {str(e)}")

# Close the chat input container
    st.markdown("</div>", unsafe_allow_html=True)



    # Sidebar with information
    # Sidebar with information
    with st.sidebar:
        st.image("images/uwgt.png", caption="Sponsor Logo", width=150)

        if st.button("Clear Chat History", key="clear_chat_sidebar"):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("""
        ### About This Assistant
        
        This knowledge assistant Chat Bot based on RAG is designed by Dr. Syed Faizan under supervision of Dr. Mudassir Azeez Khan to answer questions related to the Community Medicine Department, MMC & RI.
        
        **Course Highlights:**
        - **Housing Stability:** Trends in eviction applications, housing starts, and affordability.
        - **Financial Stability:** Income distribution, Gini coefficient trends, and living wage predictions.
        - **Demographics:** Analysis of population growth, diversity, and socio-economic factors in the Greater Toronto Area (GTA).
        - **Machine Learning Models:** Predictive analysis of low-income measures and unemployment rates.
        - **Actionable Insights:** Recommendations for stakeholders based on analytical findings.

        #### 💡 Tips for Better Results
        - Be specific in your questions.
        - Focus on project-related topics.
        - Explore housing, financial, and demographic trends in detail.
        """)

    # Enhanced footer with credits
    st.markdown("---")
    st.markdown("""
    <div style="padding: 10px; font-size: 14px; color: #444;">
        <p><strong>Team Members:</strong> Syed Faizan (Team Lead), Emelia Doku, Vraj Shah, Shicheng Wan, Pravalika Sorda, and Christiana Adjei.</p>
        <p><strong>Data Sources:</strong> Public datasets, machine learning models, and EDA outputs from the ALY 6080 Capstone Project.</p>
        <p>For additional resources or inquiries, please contact the Team Lead, Syed Faizan, via the contact section.</p>
    </div>
    """, unsafe_allow_html=True)

    # Path to the project report file
    project_report_file = "ALY_6080_Experential_learning_Group_1_Module_12_Capstone_Sponsor_Deliverable.pdf"

    # Section Header
    st.markdown("""
    <div class="section">
        <h3 style="color: #002D62; font-weight: bold;">About Our ALY 6080 Group 1 Project Deliverables</h3>
    </div>
    """, unsafe_allow_html=True)

    # Download Button for the Capstone Project Report
    with open(project_report_file, "rb") as file:
        st.download_button(
            label="📄 Click here to download our ALY 6080 Capstone Project Report",
            data=file,
            file_name="ALY_6080_Experential_learning_Group_1_Module_12_Capstone_Sponsor_Deliverable.pdf",
            mime="application/pdf",
            key="download_project_report"
        )

    # Project Highlights
    st.write("""
    - **Deliverables:** Our deliverables include:
        - Power BI Dashboards
        - Power Point and Google Slides for presentation with google drive as cloud
        - Code in R files
        - GitHub Repository
        - A Detailed Report in PDF form.
        - An Interactive RAG based App that allows users to query our report
    - **Project Focus and updates from the Mid Term presentation:** Our project has been updated on the following counts:
        - Demographics and an active Citizenry now included in our analysis along with Housing Stability, and Financial Stability in the Toronto GTA.
        - The interaction between these three critical domains is examined and is presented.
        - We have updated our models and study to incorporate all the recommendations and rectifications of our Supervisor Dr. Jay Qi, Isabel and Anh of United Way Greater Toronto.
        - The following changes have been made: 
             1. Updating the Linear Regression Models to limit predictions to 2030. 
             2. Adding Labels to all visualizations.
             3. Building our presentation around the principles of story telling with an narrative arc with an explicit beginning, middle, and and summaries.
    - **Technologies Used:**
        - **Machine Learning Predictive Algorithms in R**
        - **R** for Exploratory Data Analysis (EDA)
        - **LaTeX** for Report Formulation
        - **VSCode** as the website code IDE
        - **Langchain** and **FAISS** libraries to create the RAG backend
        - **GitHub** as version control and **OpenAI API** as LLM
        - **HTML**, **CSS** for front end, and **Streamlit** for App Development 
    """)


    # Contact Me section
    st.markdown('<div class="section"><h3>Contact the Team Lead Syed Faizan</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col2:
        st.markdown('<div class="button"><a href="https://www.linkedin.com/in/drsyedfaizanmd/">LinkedIn</a></div>', unsafe_allow_html=True)
    with col1:
        st.markdown('<a href="mailto:faizan.s@northeastern.edu">Email Syed Faizan</a>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="button"><a href="https://twitter.com/faizan_data_ml">Twitter</a></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="button"><a href="https://github.com/SYEDFAIZAN1987">GitHub</a></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    

# Dashboards and Datasets Page
elif page == "Course Materials":
    st.markdown("## Course Materials")

    # Dashboards Section
    st.markdown("### Course Materials")
    dashboards = {
        "Community Social Services Dashboard": "Dashboards/Community_Social_Services_Dashboard.pbix",
        "Demographics Dashboard": "Dashboards/DemographicsDashboard.pbix",
        "Engaged Residents Dashboard": "Dashboards/Engaged_Residents_Dashboard.pbix",
        "Financial Stability Dashboard": "Dashboards/Financial_Stability_Dashboard.pbix",
        "Housing Stability Dashboard": "Dashboards/Housing_Stability_Dashboard.pbix"
    }

    for idx, (name, path) in enumerate(dashboards.items()):  # idx is defined here
        if os.path.exists(path):  # Check if the file exists
            with open(path, "rb") as file:  # Open the file in binary mode
                st.download_button(
                    label=f"📊 Download {name}",  # Dynamic label
                    data=file,  # File data
                    file_name=os.path.basename(path),  # File name for download
                    mime="application/octet-stream",  # MIME type
                    key=f"download_button_{idx}"  # Unique key using idx
                )


    # Datasets Section
    st.markdown("### Course Materials")
    datasets = {
        "Demographics Data": "Consolidated_Datasets/combineddemographicsdata.xlsx",
        "Community Services Data": "Consolidated_Datasets/consolidatedcommunityservicesdata.xlsx",
        "Engaged Citizenry Data": "Consolidated_Datasets/consolidatedengagedcitizenry.xlsx",
        "Financial Data": "Consolidated_Datasets/consolidatedfinancialdata.xlsx",
        "Housing Data": "Consolidated_Datasets/consolidatedhousing.xlsx"
    }

    for idx, (name, path) in enumerate(datasets.items()):
        if os.path.exists(path):
            with open(path, "rb") as file:
                st.download_button(
                    label=f"📂 Download {name}",
                    data=file,
                    file_name=os.path.basename(path),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dataset_button_{idx}"
                )

# Presentation Page
elif page == "Project Details":
    st.markdown("## Project Details")

    presentation_file = "Presentation/ALY_6080_Final_SponsorProject_Group1.pptx"
    if os.path.exists(presentation_file):
        with open(presentation_file, "rb") as file:
            st.download_button(
                label="📂 Download the Final Presentation",
                data=file,
                file_name=os.path.basename(presentation_file),
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                key="download_demographic"
            )

    # Embed Google Slides Presentation
    google_slides_embed_url = "https://docs.google.com/presentation/d/e/2PACX-1vTXt9jxZXgzDgL8J_k8CjjKOmRhDlblAZyn479tJLMEuUwMv3ZgRn2OZjPfyUpiUQ/embed?start=true&loop=true&delayms=3000"

    st.markdown(
        f"""
        <iframe src="{google_slides_embed_url}" frameborder="0" width="100%" height="600px" 
        allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>
        """,
        unsafe_allow_html=True,
    )

     # Embed Google Slides Presentation
    google_slides_embed_url2 = "https://docs.google.com/presentation/d/e/2PACX-1vTnS13eSSJemdaSnQdy7nY4toi_2oF34DdNXFPKE_zcTaiarnkNON2B3_jBXGMr0Q/embed?start=true&loop=true&delayms=30000"

    st.markdown(
        f"""
        <iframe src="{google_slides_embed_url2}" frameborder="0" width="640" height="389" 
        allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>
        """,
        unsafe_allow_html=True,
    )



# Sidebar Details
with st.sidebar:
    st.image("images/uwgt.png", caption="Sponsor Logo", width=150)
    st.markdown("""
    ### About the Project
    **Project Highlights:**
    -This Project uses a RAG enhanced Chatbot as an educational aid
    - in a Community Medicine Department course
    """)
    if st.button("Clear Chat History",key="example_sidebar"):
        st.session_state.chat_history = []
