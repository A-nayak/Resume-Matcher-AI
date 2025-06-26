import pandas as pd
from typing import List, Dict
import spacy

nlp = spacy.load("en_core_web_sm")

class SkillSuggester:
    def __init__(self):
        self.skill_db = self._load_skill_database()
    
    def _load_skill_database(self) -> pd.DataFrame:
        """Load skill database (can be replaced with your own data)"""
        skills = {
            'Data Science': ['python', 'r', 'sql', 'machine learning', 'statistics'],
            'Web Development': ['javascript', 'html', 'css', 'react', 'node.js'],
            'DevOps': ['aws', 'docker', 'kubernetes', 'ci/cd', 'terraform'],
            'Mobile Development': ['swift', 'kotlin', 'flutter', 'react native']
        }
        return pd.DataFrame([(field, skill) for field in skills for skill in skills[field]], 
                          columns=['field', 'skill'])

    def suggest_skills(self, resume_data: Dict, job_description: str) -> Dict:
        """Suggest skills based on resume and job description"""
        current_skills = set(resume_data.get('skills', []))
        jd_skills = self._extract_skills_from_jd(job_description)
        
        # Find missing skills
        missing_skills = jd_skills - current_skills
        
        # Recommend similar skills from database
        recommendations = {}
        for skill in missing_skills:
            similar = self._find_similar_skills(skill)
            if similar:
                recommendations[skill] = similar
        
        return {
            "missing_skills": list(missing_skills),
            "recommendations": recommendations,
            "current_skills": list(current_skills),
            "jd_skills": list(jd_skills)
        }

    def _extract_skills_from_jd(self, jd_text: str) -> set:
        """Extract skills from job description"""
        doc = nlp(jd_text.lower())
        return {ent.text for ent in doc.ents if ent.label_ == "SKILL"}

    def _find_similar_skills(self, skill: str) -> List[str]:
        """Find similar skills from database"""
        skill = skill.lower()
        if skill in self.skill_db['skill'].values:
            field = self.skill_db[self.skill_db['skill'] == skill]['field'].iloc[0]
            return self.skill_db[self.skill_db['field'] == field]['skill'].tolist()
        return []
