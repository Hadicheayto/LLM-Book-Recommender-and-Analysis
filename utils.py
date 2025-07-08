import streamlit as st
import requests

@st.cache_data(show_spinner=False)
def get_book_cover_url(title):
    url = f"https://openlibrary.org/search.json?title={title}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['docs']:
            cover_id = data['docs'][0].get('cover_i')
            if cover_id:
                return f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
    return None
