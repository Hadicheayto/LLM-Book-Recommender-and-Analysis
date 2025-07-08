import streamlit as st
from chains.book import BookRecommender  
from utils import get_book_cover_url
from rag.rag_utils import get_vectorstore_and_BM25, get_response
from langchain_core.messages import AIMessage, HumanMessage

st.set_page_config(layout="wide", page_title="📚 LLM Book Genie", page_icon="📖")

st.markdown(
    """
        <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
        }
        .stApp {
            background-image: url("https://w0.peakpx.com/wallpaper/339/896/HD-wallpaper-library-concepts-stack-of-books-education-background-books-background-with-books-books-on-the-table.jpg");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-position: center;
        }
        .title-box {
            background-color: rgba(255, 255, 255, 0.9);
            padding: 20px;
            border-radius: 1rem;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        .title-box h1 {
            font-size: 3rem;
            margin-bottom: 0.2rem;
        }
        .title-box h4 {
            color: #555;
            margin-top: 0;
            font-weight: 400;
        }
        .book-card {
            padding: 7px;
            border-radius: 0.75rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
            width: 420px;
            height: 225px;
            /* Keep block styling */
        }
        .recommend-info {
            text-align: center;
            font-size: 18px;
            color: #444;
            #background-color: rgba(255,255,255,0.8);
            padding: 1rem;
            border-radius: 10px;
            max-width: 700px;
            margin: auto;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<div class="title-box">
    <h1>LLM Book Genie 📚</h1>
    <h4>Your smart AI assistant for book recommendations, summaries, and more ✨</h4>
</div>
""", unsafe_allow_html=True)


mode = st.sidebar.radio("Select an option:", ["📚 Recommend a Book", "📄 Analyze a Book"])

if "last_mode" not in st.session_state:
    st.session_state.last_mode = mode
if st.session_state.last_mode != mode:
    st.session_state.last_mode = mode
    st.session_state.recommend_submitted = False
    st.session_state.recommend_clicked = False

if "recommend_clicked" not in st.session_state:
    st.session_state.recommend_clicked = False

# ---------------------------
# 📚 Recommend a Book Section
# ---------------------------

if mode == "📚 Recommend a Book":

    genre = st.sidebar.selectbox(
        "Choose your favorite genre:",
        ['Science Fiction', 'Mystery', 'Romance', 'Non-fiction', 'Fantasy', 'Thriller', 'Historical', 'Self-Help']
    )

    fiction_type = st.sidebar.selectbox(
        "Do you prefer fiction or non-fiction?",
        ['Fiction', 'Non-fiction']
    )

    recent_book = st.sidebar.text_input("What’s a book you loved recently?")
    goal = st.sidebar.selectbox("Are you looking to learn or relax?", ['Learn', 'Relax'])
    submit = st.sidebar.button("✨ Get Recommendations")

    if submit and recent_book.strip():
        st.session_state.recommend_clicked = True

    if not st.session_state.recommend_clicked:
        st.markdown("""
        <div class="recommend-info">
            Welcome! Select your favorite genre and preferences from the sidebar, then click <b>✨ Get Recommendations</b> to discover your next great read.
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.recommend_clicked:
        if not recent_book.strip():
            st.error("Please enter a book you loved recently to get better recommendations!")
        else:
            with st.spinner("🧠 Thinking hard to recommend the best books for you..."):
                recommender = BookRecommender(genre, fiction_type, recent_book, goal)
                response = recommender.recommend_books()

                st.subheader("📖 Your Reader Profile")
                st.success(response['reader_profile'])

                st.subheader("📚 Book Recommendations")
                lines = response['book_list'].strip().split('\n')
                cleaned_lines = [line.strip().lstrip("-•*1234567890. ").strip() for line in lines if line.strip()]
                book_pairs = [cleaned_lines[i:i+2] for i in range(0, len(cleaned_lines), 2)]

                for pair in book_pairs:
                    cols = st.columns(2)
                    for idx, line in enumerate(pair):
                        if " by " in line and " - " in line:
                            try:
                                title_author, description = line.split(" - ", 1)
                                title, author = title_author.split(" by ")
                                title = title.strip('"“”').strip()
                                author = author.strip()
                                description = description.strip()

                                with cols[idx]:
                                    with st.container():
                                        cover_url = get_book_cover_url(title)
                                        st.markdown(
                                            f"""
                                            <div class='book-card' style='display: flex; flex-direction: row; gap: 12px; text-align: left; border-radius: 0.75rem; height: 100%; overflow: hidden;'>
                                                <div style='flex-shrink: 0;'>
                                                    {"<img src='" + cover_url + "' width='130' style='border-radius: 0.75rem; height: 100%; object-fit: cover;'/>" if cover_url else ""}
                                                </div>
                                                <div style='flex: 1; display: flex; flex-direction: column; justify-content: space-between;'>
                                                    <div>
                                                        <div style='font-weight:bold; font-size:18px; line-height: 1.2;'>{title}</div>
                                                        <div style='margin-top: 4px; font-size: 14px;'>{description}</div>
                                                    </div>
                                                    <div style='color:#555; font-style: italic; margin-top: 10px; font-size: 13px;'>by {author}</div>
                                                </div>
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )
                            except Exception as e:
                                st.warning(f"Couldn't parse recommendation: {line}")

                videos = recommender.youtube_videos_by_each_preference()
                if videos:
                    st.subheader("🎥 YouTube Videos Based on Your Preferences")
                    video_cols = st.columns(len(videos))
                    for i, video_url in enumerate(videos):
                        with video_cols[i]:
                            st.video(video_url)
                else:
                    st.info("No related YouTube videos found for your preferences.")

# ---------------------------
# 📄 Analyse a Book Section
# ---------------------------

elif mode == "📄 Analyze a Book":

    summary_type = st.sidebar.selectbox("Choose summary type:", ["Brief", "Detailed"])
    uploaded_file = st.sidebar.file_uploader("Upload your PDFs here and click on 'Process'", type=["pdf"])

    col1, col2 = st.sidebar.columns(2)
    summarize_clicked = col1.button("Summary")
    process_clicked = col2.button("Process")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "bm25" not in st.session_state:
        st.session_state.bm25 = None

    if len(st.session_state.chat_history) == 0:
        with st.chat_message("assistant"):
            st.markdown("Hello, I'm a bot. How can I help you today?")

    if summarize_clicked:
        if uploaded_file:
            with st.spinner("Generating summary..."):
                book = BookRecommender("N/A", "N/A", "N/A", "N/A")  # dummy since we only need summarize_pdf
                summary = book.summarize_pdf(uploaded_file, summary_type)
                with st.chat_message("assistant"):
                    st.markdown(f"**📄 Summary ({summary_type})**\n\n{summary}")
                #st.session_state.chat_history.append(AIMessage(content=summary))
        else:
            st.warning("Please upload a PDF first.")

    if process_clicked:
        if uploaded_file:
            with st.spinner("Processing PDF for chatting..."):
                vectorstore, bm25, _ = get_vectorstore_and_BM25([uploaded_file])
                st.session_state.vectorstore = vectorstore
                st.session_state.bm25 = bm25
                st.success("PDF processed. You can now ask questions!")
        else:
            st.warning("Please upload a PDF before processing.")

    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").markdown(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").markdown(msg.content)

    user_input = st.chat_input("Type your message here ...")
    if user_input:
        if st.session_state.vectorstore is None or st.session_state.bm25 is None:
            st.warning("Please process the PDFs.")
        else:
            st.chat_message("user").markdown(user_input)
            st.session_state.chat_history.append(HumanMessage(content=user_input))

            with st.chat_message("assistant"):
                response, context = get_response(
                    user_input,
                    st.session_state.chat_history,
                    st.session_state.vectorstore,
                    st.session_state.bm25
                )
                st.markdown(response)

            st.session_state.chat_history.append(AIMessage(content=response))










