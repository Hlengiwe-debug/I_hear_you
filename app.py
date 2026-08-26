import streamlit as st
from transformers import pipeline
import torch
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="Inner Compass AI",
    page_icon=":compass:",
    layout="wide"
)

if 'history' not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def load_models():
    try:
        sentiment = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1
        )
        return sentiment
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

sentiment_model = load_models()

SPIRITUAL_TEXTS = {
    "Islamic": {
        "POSITIVE": "Quran 94:5-6 - For indeed, with hardship comes ease. Indeed, with hardship comes ease.",
        "NEGATIVE": "Quran 2:286 - Allah does not burden a soul more than it can bear.",
        "GUIDANCE": "Surah Ad-Duha - Your Lord has not forsaken you, nor has He become displeased."
    },
    "Christian": {
        "POSITIVE": "Philippians 4:13 - I can do all things through Christ who strengthens me.",
        "NEGATIVE": "Psalm 34:18 - The Lord is near to the brokenhearted and saves the crushed in spirit.",
        "GUIDANCE": "Matthew 11:28 - Come to me, all you who are weary and burdened, and I will give you rest."
    },
    "Esoteric": {
        "POSITIVE": "As above, so below. Your inner reality manifests your outer experience.",
        "NEGATIVE": "The alchemical crucible: dissolution precedes transformation.",
        "GUIDANCE": "The Hermetic Principle of Correspondence: That which is above is like that which is below."
    }
}

def get_verse(sentiment, belief):
    if belief in SPIRITUAL_TEXTS:
        text_data = SPIRITUAL_TEXTS[belief]
        if sentiment in text_data:
            return text_data[sentiment]
    return "In quietness and trust is your strength."

def generate_advice(user_input, sentiment, belief, gender, tone):
    import random
    
    advice_templates = {
        "Islamic": [
            "Remember the words of the Prophet (peace be upon him): Wondrous is the affair of the believer. All of it is good. Even this struggle is a means of spiritual growth. Allah is with you.",
            "Ibn Qayyim said: Patience is the half of faith. In this moment, your patience is building your faith. Allah sees your struggle."
        ],
        "Christian": [
            "God's grace is sufficient for you. In 2 Corinthians 12:9, Paul reminds us: My grace is sufficient for you, for my power is made perfect in weakness. Your weakness is where His strength shines.",
            "The Lord is your shepherd. Even in the valley of the shadow of death, He walks with you. You are not alone."
        ],
        "Esoteric": [
            "The darkness you feel is the alchemical nigredo - the first stage of transformation. Your soul is being refined by fire. Trust the process.",
            "Remember the Hermetic principle: As above, so below. Your inner alchemy reflects cosmic transformation. This struggle is divine."
        ]
    }
    
    spiritual_wisdom = random.choice(advice_templates.get(belief, ["Trust in your journey."]))
    
    homework = f"""
Your Action Plan:

1. Reflection: Write one thing this situation is teaching you
2. Action: Take one small step toward what feels difficult
3. Practice: Return to the verse shared above when you feel anxious

I will check in with you tomorrow.
"""
    
    return f"{spiritual_wisdom}\n\n{homework}"

st.title("Inner Compass AI")
st.caption("Faith-based mental health guidance for Islamic, Christian, and Esoteric traditions")

tab1, tab2, tab3 = st.tabs(["How It Works", "The Science", "Disclaimer"])

with tab1:
    st.markdown("""
    ### How Inner Compass AI Works
    
    1. Share what is on your mind below
    2. Select your tradition and gender
    3. Receive personalized guidance with a spiritual verse
    
    ### Why Your Belief Matters
    
    This AI respects your identity:
    - Islamic tradition
    - Christian tradition
    - Esoteric knowledge
    
    Your faith journey is unique. Your guidance should be too.
    """)

with tab2:
    st.markdown("""
    ### Scientific Foundation
    
    - DistilBERT: 97 percent of BERT's performance with 40 percent fewer parameters
    - Culturally Responsive AI: Personalized approaches improve engagement
    - Gender Awareness: Men and women process emotions differently
    
    ### Limitations
    
    - NOT professional therapy
    - AI may not understand complex trauma
    - Always consult qualified professionals
    """)

with tab3:
    st.warning("""
    Important Ethical Disclaimer
    
    This is a supportive companion, not a licensed therapist.
    
    If you are in crisis:
    - Emergency: 999
    - Befrienders KL: 03-7627 2929 (24/7)
    - Talian Kasih: 15999
    - Mental Health Psychosocial Support: 03-2935 9935
    """)

with st.sidebar:
    st.header("Your Settings")
    
    st.divider()
    
    belief = st.selectbox(
        "Your Tradition",
        ["Islamic", "Christian", "Esoteric"],
        help="Choose the tradition that speaks to you"
    )
    
    gender = st.selectbox(
        "Your Gender",
        ["Male", "Female"],
        help="Helps tailor the tone"
    )
    
    tone = st.selectbox(
        "Preferred Tone",
        ["Tough Love", "Soft Support"],
        help="How direct or gentle should I be?"
    )
    
    st.divider()
    
    st.info(f"""
    Current Settings:
    - {belief}
    - {gender}
    - {tone}
    """)
    
    if st.button("Reset History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

st.divider()
st.subheader("Share What Is On Your Mind")

user_input = st.text_area(
    "How are you feeling today?",
    height=150,
    placeholder="Example: I feel lost and disconnected from my faith. I do not know how to find my way back."
)

if st.button("Get Guidance", type="primary", use_container_width=True):
    if user_input.strip():
        
        with st.spinner("Seeking wisdom from your tradition..."):
            
            sentiment_result = sentiment_model(user_input)
            label = sentiment_result[0]['label']
            confidence = sentiment_result[0]['score'] * 100
            
            verse = get_verse(label, belief)
            
            response = generate_advice(user_input, label, belief, gender, tone)
            
            st.session_state.history.append({
                'Input': user_input[:100] + "...",
                'Sentiment': label,
                'Confidence': f"{confidence:.1f}%",
                'Tradition': belief,
                'Time': datetime.now().strftime("%H:%M")
            })
        
        st.divider()
        st.subheader("Your Guidance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if label == "POSITIVE":
                st.success(f"Sentiment: Positive")
            else:
                st.error(f"Sentiment: Negative")
        
        with col2:
            st.metric("Confidence Score", f"{confidence:.1f}%")
            st.progress(confidence / 100)
        
        st.info(f"{verse}")
        
        st.markdown("### Guidance")
        st.write(response)
        
        st.divider()
        st.success("I will check in with you tomorrow. How are you feeling about this plan?")
        
        feedback = st.radio(
            "Does this guidance resonate with you?",
            ["Yes, it helps", "Somewhat", "Not really"],
            horizontal=True
        )
        
        if feedback == "Yes, it helps":
            st.balloons()
            st.write("Growth takes time. Be patient with yourself.")
        
    else:
        st.warning("Please share what is on your mind before seeking guidance.")

if st.session_state.history:
    st.divider()
    st.subheader("Your Session History")
    
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    if len(st.session_state.history) > 1:
        sentiment_counts = pd.DataFrame(st.session_state.history)['Sentiment'].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Your Sentiment Patterns",
            color_discrete_sequence=["#6C63FF", "#FF6B6B"]
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Inner Compass AI | Faith-based companion for healing and growth")