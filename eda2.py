import json, numpy as np
from collections import Counter
from datetime import date, datetime

path = r'c:\Users\hp\Desktop\hack2Skilll\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl'

ai_titles = {'ML Engineer','AI Engineer','Data Scientist','Senior Machine Learning Engineer',
             'Junior ML Engineer','NLP Engineer','Research Scientist','Software Engineer',
             'Backend Engineer','Full Stack Developer','DevOps Engineer','Cloud Engineer'}

ai_skills_set = {
    'pytorch','tensorflow','python','nlp','llm','embeddings','transformers',
    'rag','fine-tuning llms','sentence transformers','faiss','milvus','qdrant',
    'pinecone','weaviate','machine learning','deep learning','neural networks',
    'scikit-learn','hugging face','lora','bert','gpt','ranking','xgboost',
    'weights & biases','recommendation','search','retrieval','information retrieval',
    'tts','gans','image classification','speech recognition','object detection',
    'statistical modeling','apache spark','mlops','vector'
}

# Companies considered product-focused (NOT pure IT services)
services_only = {'TCS', 'Infosys', 'Wipro', 'Accenture', 'Cognizant', 'Capgemini'}
product_cos = {'Swiggy', 'Razorpay', 'CRED', 'Pied Piper', 'Hooli',
               'Globex Inc', 'Wayne Enterprises', 'Stark Industries',
               'Acme Corp', 'Dunder Mifflin', 'Initech'}

ai_candidates = []
today = date(2026, 6, 17)

with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        c = json.loads(line)

        title = c['profile']['current_title']
        skill_names_lower = {s['name'].lower() for s in c['skills']}
        ai_skill_match = skill_names_lower & ai_skills_set

        if len(ai_skill_match) >= 4 or title in ai_titles:
            sig = c['redrob_signals']
            yoe = c['profile']['years_of_experience']
            companies = {r['company'] for r in c['career_history']}
            non_services = companies - services_only
            services_ratio = (len(companies) - len(non_services)) / max(len(companies), 1)
            has_product = bool(companies & product_cos)

            last_active_date = datetime.strptime(sig['last_active_date'], '%Y-%m-%d').date()
            days_inactive = (today - last_active_date).days

            ai_candidates.append({
                'id': c['candidate_id'],
                'title': title,
                'yoe': yoe,
                'country': c['profile']['country'],
                'ai_skill_count': len(ai_skill_match),
                'open_to_work': sig['open_to_work_flag'],
                'response_rate': sig['recruiter_response_rate'],
                'days_inactive': days_inactive,
                'notice': sig['notice_period_days'],
                'github_score': sig['github_activity_score'],
                'has_product': has_product,
                'services_ratio': services_ratio,
                'completeness': sig['profile_completeness_score'],
                'interview_completion': sig['interview_completion_rate'],
            })

print(f'AI-relevant candidates (>=4 AI skills OR AI title): {len(ai_candidates)}')

tc = Counter(c['title'] for c in ai_candidates)
print('\nTop 20 titles in AI-relevant pool:')
for t, n in tc.most_common(20):
    print(f'  {t}: {n}')

target_yoe = [c for c in ai_candidates if 4 <= c['yoe'] <= 9]
print(f'\nIn 4-9 YoE range: {len(target_yoe)}')

available = [c for c in target_yoe
             if c['open_to_work'] and c['days_inactive'] <= 90]
print(f'Available (open_to_work + active<90d): {len(available)}')

good_github = [c for c in available if c['github_score'] >= 30]
print(f'Also good GitHub (>=30): {len(good_github)}')

india_avail = [c for c in available if c['country'] == 'India']
print(f'India-based available: {len(india_avail)}')

product_exp = [c for c in ai_candidates if c['has_product']]
print(f'\nHas product company experience: {len(product_exp)}')

pure_services = [c for c in ai_candidates if c['services_ratio'] == 1.0]
print(f'Pure services background (100%): {len(pure_services)}')

print(f'\nAvg YoE of AI-relevant: {np.mean([c["yoe"] for c in ai_candidates]):.1f}')
print(f'Avg response rate: {np.mean([c["response_rate"] for c in ai_candidates]):.2f}')
print(f'Avg notice period: {np.mean([c["notice"] for c in ai_candidates]):.1f} days')
print(f'Avg AI skill count: {np.mean([c["ai_skill_count"] for c in ai_candidates]):.1f}')

# Percentiles
yoe_vals = sorted([c['yoe'] for c in ai_candidates])
print(f'\nYoE percentiles (AI pool):')
for p in [25, 50, 75, 90]:
    idx = int(p/100 * len(yoe_vals))
    print(f'  P{p}: {yoe_vals[idx]:.1f}')

# Country breakdown for AI pool
country_counts = Counter(c['country'] for c in ai_candidates)
print('\nCountry breakdown (AI-relevant):')
for co, n in country_counts.most_common(8):
    print(f'  {co}: {n}')

# How many have short notice (<= 30 days)
short_notice = sum(1 for c in ai_candidates if c['notice'] <= 30)
print(f'\nShort notice (<=30d): {short_notice} ({short_notice/len(ai_candidates)*100:.1f}%)')
mid_notice = sum(1 for c in ai_candidates if 30 < c['notice'] <= 60)
print(f'Mid notice (31-60d): {mid_notice} ({mid_notice/len(ai_candidates)*100:.1f}%)')
long_notice = sum(1 for c in ai_candidates if c['notice'] > 90)
print(f'Long notice (>90d): {long_notice} ({long_notice/len(ai_candidates)*100:.1f}%)')
