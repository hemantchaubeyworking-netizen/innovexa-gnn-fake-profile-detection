"""
INNOVEXA Pydantic Validation Schemas
"""

from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any


class ProfileInputSchema(BaseModel):
    profile_pic: int = Field(default=1, ge=0, le=1, description="Whether the profile has a picture (1=Yes, 0=No)")
    nums_length_username: float = Field(default=0.0, ge=0.0, le=1.0, description="Ratio of numbers in username")
    fullname_words: int = Field(default=1, ge=0, description="Count of words in full name")
    nums_length_fullname: float = Field(default=0.0, ge=0.0, le=1.0, description="Ratio of numbers in full name")
    name_equals_username: int = Field(default=0, ge=0, le=1, description="Whether username equals full name (1=Yes, 0=No)")
    description_length: int = Field(default=20, ge=0, description="Character count in biography/description")
    external_url: int = Field(default=0, ge=0, le=1, description="Presence of external URL link (1=Yes, 0=No)")
    is_private: int = Field(default=0, ge=0, le=1, description="Account privacy status (1=Private, 0=Public)")
    posts_count: int = Field(default=10, ge=0, description="Total media posts")
    followers_count: int = Field(default=150, ge=0, description="Follower count")
    follows_count: int = Field(default=200, ge=0, description="Following count")
    followers_following_ratio: Optional[float] = Field(default=None, description="Ratio of followers to following")
    model_name: Optional[str] = Field(default="GraphSAGE", description="Model architecture to use for detection")

    @model_validator(mode='before')
    @classmethod
    def handle_raw_column_names(cls, data: Any):
        if not isinstance(data, dict):
            return data
            
        mapping = {
            'profile pic': 'profile_pic',
            'nums/length username': 'nums_length_username',
            'fullname words': 'fullname_words',
            'nums/length fullname': 'nums_length_fullname',
            'name==username': 'name_equals_username',
            'description length': 'description_length',
            'external URL': 'external_url',
            'private': 'is_private',
            '#posts': 'posts_count',
            '#followers': 'followers_count',
            '#follows': 'follows_count',
            'followers_following_ratio': 'followers_following_ratio',
            'model_name': 'model_name'
        }
        
        normalized = {}
        for k, v in data.items():
            mapped_key = mapping.get(k, k)
            normalized[mapped_key] = v
            
        # Compute ratio if absent
        if normalized.get('followers_following_ratio') is None:
            followers = float(normalized.get('followers_count', normalized.get('#followers', 0)))
            follows = float(normalized.get('follows_count', normalized.get('#follows', 0)))
            normalized['followers_following_ratio'] = followers / max(follows, 1.0)
            
        return normalized

    def to_dataset_features(self) -> Dict[str, Any]:
        """Converts validated schema into dictionary matching exact dataset feature names."""
        ratio = self.followers_following_ratio
        if ratio is None:
            ratio = float(self.followers_count) / max(float(self.follows_count), 1.0)
            
        return {
            'profile pic': int(self.profile_pic),
            'nums/length username': float(self.nums_length_username),
            'fullname words': int(self.fullname_words),
            'nums/length fullname': float(self.nums_length_fullname),
            'name==username': int(self.name_equals_username),
            'description length': int(self.description_length),
            'external URL': int(self.external_url),
            'private': int(self.is_private),
            '#posts': int(self.posts_count),
            '#followers': int(self.followers_count),
            '#follows': int(self.follows_count),
            'followers_following_ratio': float(ratio)
        }


class NeighborNodeSchema(BaseModel):
    neighbor_id: int
    similarity_score: float
    rank: int


class NeighborhoodContextSchema(BaseModel):
    graph_type: str
    nearest_neighbors_count: int
    neighbors: List[NeighborNodeSchema]
    note: str


class PredictionResponseSchema(BaseModel):
    id: Optional[int] = None
    predicted_class: str
    is_fake: bool
    fake_probability: float
    genuine_probability: float
    risk_category: str
    decision_threshold: float
    model_used: str
    neighborhood_context: NeighborhoodContextSchema
    disclaimer: str
    timestamp: Optional[str] = None
