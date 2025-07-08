import os
import re
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_community.llms import OpenAI
from langchain_community.tools import YouTubeSearchTool
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import tempfile  

load_dotenv()

class BookRecommender:
    def __init__(self, genre, fiction_type, recent_book, goal) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")

        os.environ['OPENAI_API_KEY'] = api_key
        self.llm = OpenAI(temperature=0.7, max_tokens=512)
        self.youtube_tool = YouTubeSearchTool()
        self.genre = genre
        self.fiction_type = fiction_type
        self.recent_book = recent_book
        self.goal = goal

    def recommend_books(self):
        # Chain 1
        prompt1 = PromptTemplate(
            input_variables=['genre', 'fiction_type', 'recent_book', 'goal'],
            template=(
                "Based on the user's preference for {fiction_type} books, "
                "genre {genre}, a recent book they enjoyed: '{recent_book}', "
                "and their current goal: '{goal}', Briefly summarize their reader profile in 2-3 sentences."
            )
        )
        reader_profile_chain = LLMChain(llm=self.llm, prompt=prompt1, output_key="reader_profile")

        # Chain 2
        prompt2 = PromptTemplate(
            input_variables=['reader_profile'],
            template=(
                "Based on this reader profile: {reader_profile}, "
                "suggest 4 great books in a bulleted list. "
                "Each bullet should be one line: **Title** by Author - short description (max 10 words)."
            )
        )
        book_recommendation_chain = LLMChain(llm=self.llm, prompt=prompt2, output_key="book_list")

        # Sequential chain to execute both
        chain = SequentialChain(
            chains=[reader_profile_chain, book_recommendation_chain],
            input_variables=['genre', 'fiction_type', 'recent_book', 'goal'],
            output_variables=['reader_profile', 'book_list']
        )

        response = chain({
            'genre': self.genre,
            'fiction_type': self.fiction_type,
            'recent_book': self.recent_book,
            'goal': self.goal
        })
        return response

    def youtube_videos_by_each_preference(self):
        videos = []

        genre_query = f"{self.genre} books"
        result1 = self.youtube_tool.run(genre_query)
        urls1 = re.findall(r'watch\?v=[\w\-]+', result1)
        if urls1:
            videos.append("https://www.youtube.com/" + urls1[0])

        fiction_query = f"why read {self.fiction_type.lower()} books"
        result2 = self.youtube_tool.run(fiction_query)
        urls2 = re.findall(r'watch\?v=[\w\-]+', result2)
        if urls2:
            videos.append("https://www.youtube.com/" + urls2[0])

        goal_query = f"books to {self.goal.lower()}"
        result3 = self.youtube_tool.run(goal_query)
        urls3 = re.findall(r'watch\?v=[\w\-]+', result3)
        if urls3:
            videos.append("https://www.youtube.com/" + urls3[0])

        return videos

    def summarize_pdf(self, file_obj, summary_type="Brief", progress_callback=None):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_obj.read())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load()

        if summary_type == "Brief":
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=300)
            chain_type = "map_reduce"
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
            chain_type = "refine"

        docs = text_splitter.split_documents(pages)

        if progress_callback:
            progress_callback(10)

        chain = load_summarize_chain(self.llm, chain_type=chain_type, verbose=False)

        if progress_callback:
            progress_callback(30)

        summary = chain.run(docs)

        if progress_callback:
            progress_callback(100)

        return summary
    
    
    

    

