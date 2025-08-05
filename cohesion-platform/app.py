import re
import yaml
from pathlib import Path
from typing import Dict, List, Any

import kss
import streamlit as st
import plotly.graph_objects as go
from soynlp.tokenizer import LTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from PyPDF2 import PdfReader

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="말이끎(Speech Flow) 발표문 피드백 보조 프로그램", page_icon="✍️", layout="wide")

# --- 모델 및 분석기, 불용어 로드 (캐시 사용) ---
@st.cache_resource
def get_tokenizer(): 
    # soynlp의 LTokenizer는 명사 추출과 유사한 효과를 냅니다.
    return LTokenizer()

@st.cache_resource
def load_stopwords(filepath):
    stopwords_path = Path(filepath)
    if stopwords_path.exists():
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            return {line.strip() for line in f}
    return set()

tokenizer = get_tokenizer()
STOPWORDS = load_stopwords("stopwords.txt")

# --- 핵심 분석 함수들 ---

def analyze_cohesion(text: str, markers: List[str]) -> Dict[str, Any]:
    """텍스트의 응집성을 분석합니다."""
    if not markers: return {"html": text.replace('\n', '<br>'), "markers_list": []}
    
    escaped_markers = [re.escape(marker) for marker in markers]
    pattern = re.compile(r'\b(' + '|'.join(escaped_markers) + r')\b')
    
    def highlight_repl(match): return f'<span class="marker">{match.group(0)}</span>'
    highlighted_text = pattern.sub(highlight_repl, text).replace('\n', '<br>')
    
    return {"html": highlighted_text, "markers_list": markers}

def soynlp_custom_tokenizer(text: str) -> List[str]:
    """soynlp 토크나이저와 불용어 필터링을 결합한 맞춤형 토크나이저입니다."""
    tokens = tokenizer.tokenize(text, flatten=True)
    return [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1]

def extract_keywords(sentences: List[str], top_n: int) -> List[str]:
    """soynlp 맞춤형 토크나이저를 사용하여 TF-IDF 기반 핵심 키워드를 추출합니다."""
    if not sentences: return []
    
    try:
        # TfidfVectorizer에 맞춤형 토크나이저를 직접 전달합니다.
        vectorizer = TfidfVectorizer(
            tokenizer=soynlp_custom_tokenizer,
            min_df=1,
        )
        
        # 원본 문장 리스트를 그대로 전달하여 분석의 일관성을 유지합니다.
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        feature_names = vectorizer.get_feature_names_out()
        total_scores = tfidf_matrix.sum(axis=0).A1
        sorted_indices = total_scores.argsort()[::-1]
        
        # 이미 토크나이저 단계에서 필터링되었으므로, 바로 상위 N개를 반환합니다.
        keywords = [feature_names[i] for i in sorted_indices]
        return keywords[:top_n]
    except ValueError: 
        return []

# --- 중심 문장 분석 로직 ---

def score_all_sentences(paragraph: str, main_topic_list: List[str], markers: List[str], mode: str, config: Dict) -> Dict:
    """한 문단 내 모든 문장의 점수를 계산하고 상세 내역과 문단 핵심어를 반환합니다."""
    sentences = kss.split_sentences(paragraph)
    if not sentences: return {"scored_sentences": [], "p_keywords": []}
        
    p_keywords = extract_keywords(sentences, 3)
    scored_sentences = []

    if mode == "soft":
        weights = config
        for s in sentences:
            score_breakdown = {'주제어': 0.0, '문단 핵심어': 0.0, '담화 표지': 0.0}
            reasons = []
            
            if weights.get('use_main_topic', True) and main_topic_list:
                found_topics = [topic for topic in main_topic_list if topic in s]
                if found_topics:
                    score_breakdown['주제어'] = weights.get('main_topic', 1.0) * len(found_topics)
                    reasons.append(f"주제어({', '.join(found_topics)})")

            if weights.get('use_p_keyword', True):
                found_kws = [kw for kw in p_keywords if kw in s]
                if found_kws:
                    score_breakdown['문단 핵심어'] = weights.get('p_keyword', 1.0) * len(found_kws)
                    reasons.append(f"문단핵심어({', '.join(found_kws)})")

            if weights.get('use_marker', True):
                found_ms = [m for m in markers if m in s]
                if found_ms:
                    score_breakdown['담화 표지'] = weights.get('marker', 0.5)
                    reasons.append(f"담화표지({', '.join(found_ms)})")
            
            total_score = sum(score_breakdown.values())
            scored_sentences.append({"sentence": s, "score": total_score, "reasons": reasons, "breakdown": score_breakdown})
    
    elif mode == "hard":
        custom_rules = config
        rule_labels = [f"규칙 {i+1}: {rule['condition']}" for i, rule in enumerate(custom_rules)]

        for s in sentences:
            breakdown = {label: 0.0 for label in rule_labels}
            
            for i, rule in enumerate(custom_rules):
                condition_met = False
                condition_type = rule['condition']
                value = rule.get('value', '')

                if condition_type == "주제어 포함":
                    if main_topic_list and any(topic in s for topic in main_topic_list): condition_met = True
                elif condition_type == "문단 핵심어 포함":
                    if any(kw in s for kw in p_keywords): condition_met = True
                elif condition_type == "담화 표지 포함":
                    if any(m in s for m in markers): condition_met = True
                elif condition_type == "특정 단어 포함":
                    if value and value in s: condition_met = True
                elif condition_type == "문장 길이 (이상)":
                    if value and len(s) >= int(value): condition_met = True
                elif condition_type == "문장 길이 (이하)":
                    if value and len(s) <= int(value): condition_met = True
                
                if condition_met:
                    action = rule['action']
                    score_value = rule.get('score_value', 1.0)
                    rule_label = rule_labels[i]
                    if action == "점수 더하기":
                        breakdown[rule_label] += score_value
                    elif action == "점수 빼기":
                        breakdown[rule_label] -= score_value

            total_score = sum(breakdown.values())
            reasons = [f"{key.split(':')[1].strip()}" for key, val in breakdown.items() if val > 0]
            scored_sentences.append({"sentence": s, "score": total_score, "reasons": reasons, "breakdown": breakdown})

    return {"scored_sentences": scored_sentences, "p_keywords": p_keywords}

# --- 메인 애플리케이션 UI ---
def main():
    st.title("✍️ Speech Flow")

    st.markdown("""
    <style>
    .marker { background: #ffeb3b80; padding: 2px 4px; border-radius: 4px; font-weight: bold; }
    .key-sentence { background: #d1e7dd; border-left: 5px solid #198754; padding: 8px 12px; margin: 10px 0; display: block; border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)

    if 'custom_rules' not in st.session_state:
        st.session_state.custom_rules = [{'condition': '주제어 포함', 'action': '점수 더하기', 'value': '', 'score_value': 1.0}]
    if 'key_sentences_for_reorder' not in st.session_state:
        st.session_state.key_sentences_for_reorder = []
    if 'soft_config' not in st.session_state:
        st.session_state.soft_config = {
            'use_main_topic': True, 'main_topic': 1.5,
            'use_p_keyword': True, 'p_keyword': 1.0,
            'use_marker': True, 'marker': 0.5
        }

    with st.sidebar:
        st.header("1️⃣ 담화 표지 목록 설정")
        default_markers = "그러나\n하지만\n따라서\n그러므로\n그리고\n한편\n즉\n예를 들어\n결론적으로\n반면에\n또한"
        markers_text = st.text_area("한 줄에 하나씩 담화 표지를 입력하세요.", value=default_markers, height=200)
        markers_list = [line.strip() for line in markers_text.split('\n') if line.strip()]

        st.divider()
        st.header("2️⃣ 중심 문장 분석 설정")
        main_topic_input = st.text_input("글의 주제어 (쉼표로 여러 개, 또는 문장 입력 가능)", placeholder="예: 인공지능, 미래").strip()
        
        main_topic_list = []
        if main_topic_input:
            if ',' in main_topic_input:
                main_topic_list = [word.strip() for word in main_topic_input.split(',') if word.strip()]
            else:
                # 문장으로 입력 시, 의미 있는 단어만 추출 (soynlp 사용)
                tokens = tokenizer.tokenize(main_topic_input, flatten=True)
                main_topic_list = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
        
        if main_topic_list:
            st.caption(f"분석에 적용될 주제어: `{', '.join(main_topic_list)}`")

        analysis_mode = st.radio(
            "분석 방식 선택",
            ["간편 설정", "규칙 빌더 (고급)"],
            captions=["간단한 가중치 조절 방식", "IF-THEN 규칙 직접 조립"]
        )
        
        if analysis_mode == "간편 설정":
            st.session_state.soft_config['use_main_topic'] = st.checkbox("주제어 포함 여부", value=st.session_state.soft_config['use_main_topic'], key="soft_main")
            st.session_state.soft_config['main_topic'] = st.number_input("주제어 가중치", min_value=0.0, value=st.session_state.soft_config['main_topic'], step=0.1, key="soft_main_w")
            st.session_state.soft_config['use_p_keyword'] = st.checkbox("문단 핵심어 포함 여부", value=st.session_state.soft_config['use_p_keyword'], key="soft_p")
            st.session_state.soft_config['p_keyword'] = st.number_input("문단 핵심어 가중치", min_value=0.0, value=st.session_state.soft_config['p_keyword'], step=0.1, key="soft_p_w")
            st.session_state.soft_config['use_marker'] = st.checkbox("담화 표지 포함 여부", value=st.session_state.soft_config['use_marker'], key="soft_m")
            st.session_state.soft_config['marker'] = st.number_input("담화 표지 가중치", min_value=0.0, value=st.session_state.soft_config['marker'], step=0.1, key="soft_m_w")

        else: # "규칙 빌더 (고급)"
            for i, rule in enumerate(st.session_state.custom_rules):
                st.markdown(f"---")
                cols = st.columns([4, 4, 1])
                with cols[0]:
                    rule['condition'] = st.selectbox("IF (조건)", ["주제어 포함", "문단 핵심어 포함", "담화 표지 포함", "특정 단어 포함", "문장 길이 (이상)", "문장 길이 (이하)"], key=f"cond_{i}")
                    if "특정 단어" in rule['condition']: rule['value'] = st.text_input("단어", key=f"val_word_{i}")
                    elif "문장 길이" in rule['condition']: rule['value'] = st.number_input("글자 수", min_value=1, value=50, key=f"val_len_{i}")
                with cols[1]:
                    rule['action'] = st.selectbox("THEN (실행)", ["점수 더하기", "점수 빼기"], key=f"act_{i}")
                    if "점수" in rule['action']: rule['score_value'] = st.number_input("점수", min_value=0.1, value=1.0, step=0.1, key=f"val_score_{i}")
                with cols[2]:
                    st.write(""); st.write("")
                    if st.button("❌", key=f"del_{i}"):
                        st.session_state.custom_rules.pop(i)
                        st.rerun()
            if st.button("➕ 새 규칙 추가"):
                st.session_state.custom_rules.append({'condition': '주제어 포함', 'action': '점수 더하기', 'value': '', 'score_value': 1.0})
                st.rerun()

    st.header("1️⃣ 텍스트 입력")
    st.info("아래에 텍스트를 직접 붙여넣거나, .txt 또는 .pdf 파일을 업로드하세요.")
    
    uploaded_file = st.file_uploader("파일 업로드", type=['txt', 'pdf'])
    
    file_content = ""
    if uploaded_file is not None:
        try:
            if uploaded_file.type == "text/plain":
                file_content = uploaded_file.getvalue().decode("utf-8")
            elif uploaded_file.type == "application/pdf":
                pdf_reader = PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    file_content += page.extract_text() + "\n"
        except Exception as e:
            st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")

    default_text = file_content if file_content else ""
    text_input = st.text_area("분석할 텍스트", value=default_text, height=300)
    
    run_button = st.button("분석 실행", type="primary")

    if run_button:
        if not text_input:
            st.warning("분석할 텍스트를 먼저 입력해주세요.")
            return

        with st.spinner("텍스트를 분석 중입니다..."):
            cohesion_result = analyze_cohesion(text_input, markers_list)
            
            st.divider()
            st.subheader("📝 담화표지 하이라이트")
            st.markdown(cohesion_result['html'], unsafe_allow_html=True)

            st.divider()
            st.subheader("📍 중심 문장 분석 결과")
            st.caption(f"분석 모드: **{analysis_mode}** | 적용된 주제어: **`{', '.join(main_topic_list)}`**")

            st.session_state.key_sentences_for_reorder = []
            
            paragraphs = [p.strip() for p in re.split(r'\n+', text_input.strip()) if p.strip()]
            
            for i, para in enumerate(paragraphs):
                mode = "soft" if analysis_mode == "간편 설정" else "hard"
                current_config = st.session_state.soft_config if mode == "soft" else st.session_state.custom_rules
                
                analysis_result = score_all_sentences(para, main_topic_list, markers_list, mode, current_config)
                all_scored_sentences = analysis_result["scored_sentences"]
                p_keywords = analysis_result["p_keywords"]
                
                st.markdown(f"**문단 {i+1}**")
                if p_keywords:
                    st.markdown(f"> **문단 핵심어:** `{'`, `'.join(p_keywords)}`")

                if all_scored_sentences:
                    best_sentence_info = max(all_scored_sentences, key=lambda x: x['score'])
                    
                    if best_sentence_info['score'] > 0:
                        st.session_state.key_sentences_for_reorder.append(best_sentence_info['sentence'])
                        highlighted_paragraph = para.replace(best_sentence_info['sentence'], f"<div class='key-sentence'>{best_sentence_info['sentence']}</div>")
                        st.markdown(highlighted_paragraph, unsafe_allow_html=True)
                        
                        breakdown = best_sentence_info['breakdown']
                        details = [f"{key.split(':')[1].strip() if ':' in key else key}({val:.1f})" for key, val in breakdown.items() if val > 0]
                        st.caption(f"선정 이유: {' + '.join(details)} (총점: {best_sentence_info['score']:.2f})")

                        with st.expander("문단 내 문장별 점수 상세 보기"):
                            for k, s_info in enumerate(all_scored_sentences):
                                s_breakdown = s_info['breakdown']
                                s_details = [f"{key.split(':')[1].strip() if ':' in key else key}({val:.1f})" for key, val in s_breakdown.items() if val > 0]
                                if not s_details:
                                    st.write(f"_{k+1}번째 문장: (점수 없음)_")
                                else:
                                    st.write(f"**{k+1}번째 문장 (총점: {s_info['score']:.2f}):** {' + '.join(s_details)}")
                            
                            st.markdown("---")
                            st.markdown("##### 문장별 점수 비교 그래프")
                            sentences_labels = [f"{j+1}번째" for j in range(len(all_scored_sentences))]
                            all_rule_names = sorted(list(set(key for s in all_scored_sentences for key in s['breakdown'].keys())))
                            breakdown_data = {name: [s['breakdown'].get(name, 0) for s in all_scored_sentences] for name in all_rule_names}
                            
                            fig = go.Figure()
                            for name, data in breakdown_data.items():
                                fig.add_trace(go.Bar(name=name, x=sentences_labels, y=data))
                            fig.update_layout(barmode='stack', title_text='문장별 점수 구성', xaxis_title="문장", yaxis_title="점수")
                            st.plotly_chart(fig, use_container_width=True)

                    else:
                        st.markdown(para)
                        st.caption("중심 문장을 찾지 못했습니다.")
                else:
                    st.markdown(para)
                    st.caption("분석할 문장이 없습니다.")
                st.markdown("---")

    if st.session_state.key_sentences_for_reorder:
        st.divider()
        st.subheader("✨ 핵심 문장 구조화")
        st.info("추출된 핵심 문장들의 순서와 역할을 재구성하여 글의 최종 뼈대를 완성하세요.")

        num_sentences = len(st.session_state.key_sentences_for_reorder)

        for i in range(num_sentences):
            cols = st.columns([2, 8, 1, 1, 1])
            
            role = ""
            if i == 0: role = "도입"
            elif i == num_sentences - 1 and num_sentences > 1: role = "마무리"
            else: role = f"전개 {i}"
            
            cols[0].markdown(f"### {role}")

            user_input = cols[1].text_area(f"문장 {i+1} 편집", value=st.session_state.key_sentences_for_reorder[i], label_visibility="collapsed", key=f"reorder_text_{i}")
            st.session_state.key_sentences_for_reorder[i] = user_input
            
            if cols[2].button("🔼", key=f"up_{i}", help="위로 이동") and i > 0:
                st.session_state.key_sentences_for_reorder.insert(i-1, st.session_state.key_sentences_for_reorder.pop(i))
                st.rerun()

            if cols[3].button("🔽", key=f"down_{i}", help="아래로 이동") and i < num_sentences - 1:
                st.session_state.key_sentences_for_reorder.insert(i+1, st.session_state.key_sentences_for_reorder.pop(i))
                st.rerun()
            
            if cols[4].button("🗑️", key=f"del_{i}", help="삭제"):
                st.session_state.key_sentences_for_reorder.pop(i)
                st.rerun()

        if st.button("➕ 새 문장 추가"):
            st.session_state.key_sentences_for_reorder.append("여기에 새로운 문장을 입력하세요.")
            st.rerun()
            
        st.markdown("---")
        
        st.subheader("결과 내보내기")
        
        final_text_lines = []
        for i, sentence in enumerate(st.session_state.key_sentences_for_reorder):
            role = ""
            if i == 0: role = "도입"
            elif i == len(st.session_state.key_sentences_for_reorder) - 1 and len(st.session_state.key_sentences_for_reorder) > 1: role = "마무리"
            else: role = f"전개 {i}"
            final_text_lines.append(f"[{role}] {sentence}")

        final_text = "\n\n".join(final_text_lines)
        
        st.text_area("복사하기", value=final_text, height=150)
        
        st.download_button(
            label="텍스트 파일로 다운로드 (.txt)",
            data=final_text.encode('utf-8'),
            file_name='structured_summary.txt',
            mime='text/plain',
        )

if __name__ == "__main__":
    main()
