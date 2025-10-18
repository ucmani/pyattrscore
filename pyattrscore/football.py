import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .base import AttributionModel, TouchpointData, AttributionConfig
from .exceptions import InsufficientDataError, AttributionCalculationError, ConfigurationError
from .utils import (
    validate_dataframe_structure,
    convert_timestamp_column,
    sort_by_timestamp,
    group_touchpoints_by_user,
    create_attribution_result_dataframe,
    safe_divide
)
from .logger import get_logger

logger = get_logger(__name__)


class ChannelArchetype(Enum):
    """Channel archetypes based on football positions"""
    GENERATOR = "generator"  # Creates awareness, starts plays
    ASSISTER = "assister"    # Nurtures and sets up conversions
    CLOSER = "closer"        # Finishes conversions
    PARTICIPANT = "participant"  # Supporting role


class FootballRole(Enum):
    """Football-inspired roles for touchpoints"""
    SCORER = "scorer"                    # Final conversion touchpoint
    ASSISTER = "assister"               # Setup touchpoint before conversion
    KEY_PASSER = "key_passer"           # Starts the conversion build-up
    MOST_PASSES = "most_passes"         # Most frequent engagement
    MOST_MINUTES = "most_minutes"       # Longest engagement time
    MOST_DRIBBLES = "most_dribbles"     # Revives cold leads
    PARTICIPANT = "participant"          # Supporting touchpoint


@dataclass
class FootballMetrics:
    """Football-inspired metrics for channels"""
    goals: int = 0                      # Total conversions
    assists: int = 0                    # Setup conversions
    key_passes: int = 0                 # Journey initiations
    passes: int = 0                     # Total touchpoints
    minutes: float = 0.0                # Total engagement time
    dribbles: int = 0                   # Lead revivals
    expected_goals: float = 0.0         # xG - conversion probability
    expected_assists: float = 0.0       # xA - assist probability
    channel_impact_score: float = 0.0   # Overall CIS


class FootballAttributionConfig(AttributionConfig):
    """Extended configuration for Football Attribution Model"""
    
    # Role weights for CIS calculation
    scorer_weight: float = 0.25
    assister_weight: float = 0.20
    key_passer_weight: float = 0.15
    most_passes_weight: float = 0.15
    most_minutes_weight: float = 0.10
    most_dribbles_weight: float = 0.10
    participant_weight: float = 0.05
    
    # Baseline weight for all touchpoints
    baseline_weight: float = 0.1
    
    # Time thresholds
    cold_lead_threshold_days: int = 7    # Days to consider a lead "cold"
    engagement_time_weight: float = 1.0  # Weight for engagement time calculation
    
    # Channel archetype mapping (can be customized)
    channel_archetypes: Optional[Dict[str, str]] = None
    
    def __init__(self, **data):
        """Initialize with validation and default values"""
        super().__init__(**data)
        
        # Validate weights sum appropriately
        total_weight = (
            self.scorer_weight + self.assister_weight + self.key_passer_weight +
            self.most_passes_weight + self.most_minutes_weight +
            self.most_dribbles_weight + self.participant_weight
        )
        
        if abs(total_weight - 1.0) > 0.001:
            raise ConfigurationError(
                f"Role weights must sum to 1.0, got {total_weight}"
            )
        
        # Set default channel archetypes if not provided
        if self.channel_archetypes is None:
            self.channel_archetypes = {
                'organic_search': 'generator',
                'paid_search': 'assister',
                'social_media': 'generator',
                'email': 'assister',
                'direct': 'closer',
                'referral': 'closer',
                'display': 'generator',
                'video': 'assister'
            }


class FootballAttribution(AttributionModel):
    """
    Football-Inspired Attribution Model
    
    This model treats each conversion as a "goal" and assigns football-inspired
    roles to marketing channels based on their position and contribution in the
    customer journey. It calculates Channel Impact Score (CIS) using weighted
    role assignments and provides football analytics metrics.
    
    Key Features:
    - Role-based attribution (Scorer, Assister, Key Passer, etc.)
    - Channel Impact Score (CIS) calculation
    - Football metrics (Goals, Assists, Expected Goals, etc.)
    - Channel archetype classification
    - Team performance analytics
    """
    
    def __init__(self, config: Optional[FootballAttributionConfig] = None):
        """
        Initialize Football Attribution model.
        
        Args:
            config: Football-specific configuration object
        """
        if config is None:
            config = FootballAttributionConfig()
        elif not isinstance(config, FootballAttributionConfig):
            # Convert regular AttributionConfig to FootballAttributionConfig
            config = FootballAttributionConfig(**config.dict())
            
        super().__init__(config)
        self.football_config = config
        logger.info(f"Initialized Football Attribution model with {self.config.attribution_window_days}-day window")
    
    def calculate_attribution(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate football-inspired attribution scores.
        
        This method assigns football roles to touchpoints and calculates
        Channel Impact Score (CIS) based on role weights and contributions.
        
        Args:
            data: Input DataFrame with touchpoint data
            
        Returns:
            DataFrame with attribution scores and football metrics
            
        Raises:
            InsufficientDataError: If there's insufficient data for calculation
            AttributionCalculationError: If calculation fails
        """
        logger.info("Starting Football Attribution calculation")
        
        try:
            # Validate input data structure
            required_columns = ['user_id', 'touchpoint_id', 'channel', 'timestamp']
            validate_dataframe_structure(data, required_columns)
            
            # Convert and sort by timestamp
            data = convert_timestamp_column(data)
            data = sort_by_timestamp(data)
            
            # Add engagement time if not present (mock calculation)
            if 'engagement_time' not in data.columns:
                data['engagement_time'] = np.random.exponential(30, len(data))  # Mock engagement time
            
            # Validate input data using Pydantic models
            touchpoints = self.validate_input_data(data)
            
            if not touchpoints:
                raise InsufficientDataError("No valid touchpoints found in input data")
            
            # Group touchpoints by user journey
            user_journeys = self.group_by_user_journey(touchpoints)
            
            # Filter to only include users with conversions
            converting_users = self._filter_converting_users(user_journeys)
            
            if not converting_users:
                raise InsufficientDataError("No converting users found in the data")
            
            logger.info(f"Processing {len(converting_users)} converting users")
            
            # Calculate attribution for each user
            all_attributions = []
            all_metrics = {}
            
            for user_id, user_touchpoints in converting_users.items():
                user_attribution, user_metrics = self._calculate_user_football_attribution(
                    user_touchpoints, data
                )
                all_attributions.extend(user_attribution)
                
                # Aggregate metrics by channel
                for channel, metrics in user_metrics.items():
                    if channel not in all_metrics:
                        all_metrics[channel] = FootballMetrics()
                    self._aggregate_metrics(all_metrics[channel], metrics)
            
            # Create result DataFrame
            if not all_attributions:
                raise AttributionCalculationError("No attributions calculated")
            
            # Convert back to DataFrame format for result creation
            result_data = []
            attribution_scores = []
            football_roles = []
            channel_archetypes = []
            
            for attribution in all_attributions:
                touchpoint_data = attribution['touchpoint'].__dict__.copy()
                # Add engagement time from original data
                original_row = data[
                    (data['user_id'] == touchpoint_data['user_id']) & 
                    (data['touchpoint_id'] == touchpoint_data['touchpoint_id'])
                ]
                if not original_row.empty:
                    touchpoint_data['engagement_time'] = original_row.iloc[0]['engagement_time']
                
                result_data.append(touchpoint_data)
                attribution_scores.append(attribution['cis_score'])
                football_roles.append(attribution['roles'])
                
                # Determine channel archetype
                channel = touchpoint_data['channel']
                archetype = self.football_config.channel_archetypes.get(
                    channel.lower(), 'participant'
                )
                channel_archetypes.append(archetype)
            
            result_df = pd.DataFrame(result_data)
            
            # Create standardized result with additional football columns
            final_result = create_attribution_result_dataframe(
                result_df, 
                attribution_scores, 
                self.model_name
            )
            
            # Add football-specific columns
            final_result['football_roles'] = football_roles
            final_result['channel_archetype'] = channel_archetypes
            
            # Add football metrics summary
            final_result = self._add_football_metrics_to_result(final_result, all_metrics)
            
            # Validate results
            self._validate_results(final_result)
            
            logger.info(f"Football Attribution completed successfully. "
                       f"Processed {len(final_result)} touchpoints across "
                       f"{len(all_metrics)} channels")
            
            return final_result
            
        except (InsufficientDataError, AttributionCalculationError, ConfigurationError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Football Attribution: {str(e)}")
            raise AttributionCalculationError(
                f"Failed to calculate Football Attribution: {str(e)}",
                model_name=self.model_name
            )
    
    def _filter_converting_users(self, user_journeys: Dict[str, List[TouchpointData]]) -> Dict[str, List[TouchpointData]]:
        """Filter user journeys to only include those with conversions."""
        converting_users = {}
        
        for user_id, touchpoints in user_journeys.items():
            has_conversion = any(tp.conversion for tp in touchpoints)
            if has_conversion:
                converting_users[user_id] = touchpoints
        
        logger.debug(f"Filtered to {len(converting_users)} converting users "
                    f"from {len(user_journeys)} total users")
        
        return converting_users
    
    def _calculate_user_football_attribution(
        self, 
        touchpoints: List[TouchpointData], 
        original_data: pd.DataFrame
    ) -> Tuple[List[Dict[str, Any]], Dict[str, FootballMetrics]]:
        """
        Calculate football attribution for a single user's journey.
        
        Args:
            touchpoints: List of touchpoints for a single user
            original_data: Original DataFrame with engagement time data
            
        Returns:
            Tuple of (attribution list, channel metrics)
        """
        if not touchpoints:
            return [], {}
        
        # Sort touchpoints by timestamp
        sorted_touchpoints = sorted(touchpoints, key=lambda x: x.timestamp)
        
        # Find conversion touchpoints
        conversion_touchpoints = [tp for tp in sorted_touchpoints if tp.conversion]
        if not conversion_touchpoints:
            return [], {}
        
        # Filter touchpoints within attribution window
        windowed_touchpoints = self.filter_by_attribution_window(sorted_touchpoints)
        if not windowed_touchpoints:
            return [], {}
        
        # Assign football roles
        role_assignments = self._assign_football_roles(windowed_touchpoints, original_data)
        
        # Calculate CIS for each touchpoint
        attributions = []
        channel_metrics = {}
        
        for touchpoint in windowed_touchpoints:
            roles = role_assignments.get(touchpoint.touchpoint_id, [])
            cis_score = self._calculate_cis_score(roles)
            
            attributions.append({
                'touchpoint': touchpoint,
                'cis_score': cis_score,
                'roles': roles
            })
            
            # Update channel metrics
            channel = touchpoint.channel
            if channel not in channel_metrics:
                channel_metrics[channel] = FootballMetrics()
            
            self._update_channel_metrics(
                channel_metrics[channel], 
                touchpoint, 
                roles, 
                cis_score,
                original_data
            )
        
        # Normalize CIS scores to sum to 1.0
        total_cis = sum(attr['cis_score'] for attr in attributions)
        if total_cis > 0:
            for attribution in attributions:
                attribution['cis_score'] = safe_divide(attribution['cis_score'], total_cis)
        
        return attributions, channel_metrics
    
    def _assign_football_roles(
        self, 
        touchpoints: List[TouchpointData], 
        original_data: pd.DataFrame
    ) -> Dict[str, List[str]]:
        """
        Assign football roles to touchpoints based on their position and characteristics.
        
        Args:
            touchpoints: List of touchpoints in the journey
            original_data: Original DataFrame with engagement time data
            
        Returns:
            Dictionary mapping touchpoint_id to list of assigned roles
        """
        role_assignments = {}
        
        if not touchpoints:
            return role_assignments
        
        # Initialize all touchpoints as participants
        for tp in touchpoints:
            role_assignments[tp.touchpoint_id] = [FootballRole.PARTICIPANT.value]
        
        # Find conversion touchpoints
        conversion_touchpoints = [tp for tp in touchpoints if tp.conversion]
        
        if conversion_touchpoints:
            # Assign Scorer role to the last conversion touchpoint
            last_conversion = max(conversion_touchpoints, key=lambda x: x.timestamp)
            role_assignments[last_conversion.touchpoint_id] = [FootballRole.SCORER.value]
            
            # Assign Assister role to the touchpoint before the last conversion
            conversion_index = touchpoints.index(last_conversion)
            if conversion_index > 0:
                assister = touchpoints[conversion_index - 1]
                if FootballRole.SCORER.value not in role_assignments[assister.touchpoint_id]:
                    role_assignments[assister.touchpoint_id] = [FootballRole.ASSISTER.value]
        
        # Assign Key Passer role to the first touchpoint
        if touchpoints:
            first_touchpoint = touchpoints[0]
            if (FootballRole.SCORER.value not in role_assignments[first_touchpoint.touchpoint_id] and
                FootballRole.ASSISTER.value not in role_assignments[first_touchpoint.touchpoint_id]):
                role_assignments[first_touchpoint.touchpoint_id] = [FootballRole.KEY_PASSER.value]
        
        # Assign Most Passes role to the channel with most touchpoints
        channel_counts = {}
        for tp in touchpoints:
            channel_counts[tp.channel] = channel_counts.get(tp.channel, 0) + 1
        
        if channel_counts:
            most_passes_channel = max(channel_counts, key=channel_counts.get)
            most_passes_touchpoints = [tp for tp in touchpoints if tp.channel == most_passes_channel]
            if most_passes_touchpoints:
                # Assign to the first touchpoint of this channel
                tp_id = most_passes_touchpoints[0].touchpoint_id
                if FootballRole.PARTICIPANT.value in role_assignments[tp_id]:
                    role_assignments[tp_id] = [FootballRole.MOST_PASSES.value]
                else:
                    role_assignments[tp_id].append(FootballRole.MOST_PASSES.value)
        
        # Assign Most Minutes role based on engagement time
        max_engagement_time = 0
        most_minutes_tp = None
        
        for tp in touchpoints:
            # Get engagement time from original data
            original_row = original_data[
                (original_data['user_id'] == tp.user_id) & 
                (original_data['touchpoint_id'] == tp.touchpoint_id)
            ]
            if not original_row.empty:
                engagement_time = original_row.iloc[0]['engagement_time']
                if engagement_time > max_engagement_time:
                    max_engagement_time = engagement_time
                    most_minutes_tp = tp
        
        if most_minutes_tp:
            tp_id = most_minutes_tp.touchpoint_id
            if FootballRole.PARTICIPANT.value in role_assignments[tp_id]:
                role_assignments[tp_id] = [FootballRole.MOST_MINUTES.value]
            else:
                role_assignments[tp_id].append(FootballRole.MOST_MINUTES.value)
        
        # Assign Most Dribbles role to touchpoints that revive cold leads
        cold_threshold = timedelta(days=self.football_config.cold_lead_threshold_days)
        
        for i in range(1, len(touchpoints)):
            current_tp = touchpoints[i]
            previous_tp = touchpoints[i-1]
            
            time_gap = current_tp.timestamp - previous_tp.timestamp
            if time_gap > cold_threshold:
                # This touchpoint revived a cold lead
                tp_id = current_tp.touchpoint_id
                if FootballRole.PARTICIPANT.value in role_assignments[tp_id]:
                    role_assignments[tp_id] = [FootballRole.MOST_DRIBBLES.value]
                else:
                    role_assignments[tp_id].append(FootballRole.MOST_DRIBBLES.value)
                break  # Only assign to the first cold lead revival
        
        return role_assignments
    
    def _calculate_cis_score(self, roles: List[str]) -> float:
        """
        Calculate Channel Impact Score (CIS) based on assigned roles.
        
        Args:
            roles: List of assigned football roles
            
        Returns:
            CIS score for the touchpoint
        """
        config = self.football_config
        
        # Start with baseline weight
        cis = config.baseline_weight
        
        # Add role-specific weights
        role_weights = {
            FootballRole.SCORER.value: config.scorer_weight,
            FootballRole.ASSISTER.value: config.assister_weight,
            FootballRole.KEY_PASSER.value: config.key_passer_weight,
            FootballRole.MOST_PASSES.value: config.most_passes_weight,
            FootballRole.MOST_MINUTES.value: config.most_minutes_weight,
            FootballRole.MOST_DRIBBLES.value: config.most_dribbles_weight,
            FootballRole.PARTICIPANT.value: config.participant_weight
        }
        
        # Calculate weighted contribution
        role_contribution = 0.0
        for role in roles:
            role_contribution += role_weights.get(role, 0.0)
        
        # Apply the formula from the specification
        cis = config.baseline_weight + (1 - config.baseline_weight) * role_contribution
        
        return cis
    
    def _update_channel_metrics(
        self, 
        metrics: FootballMetrics, 
        touchpoint: TouchpointData, 
        roles: List[str], 
        cis_score: float,
        original_data: pd.DataFrame
    ):
        """Update channel metrics based on touchpoint and roles."""
        
        # Update basic counts
        metrics.passes += 1
        metrics.channel_impact_score += cis_score
        
        # Update role-specific metrics
        if FootballRole.SCORER.value in roles:
            metrics.goals += 1
        if FootballRole.ASSISTER.value in roles:
            metrics.assists += 1
        if FootballRole.KEY_PASSER.value in roles:
            metrics.key_passes += 1
        if FootballRole.MOST_DRIBBLES.value in roles:
            metrics.dribbles += 1
        
        # Update engagement time
        original_row = original_data[
            (original_data['user_id'] == touchpoint.user_id) & 
            (original_data['touchpoint_id'] == touchpoint.touchpoint_id)
        ]
        if not original_row.empty:
            engagement_time = original_row.iloc[0]['engagement_time']
            metrics.minutes += engagement_time
        
        # Calculate expected goals and assists (simplified)
        if touchpoint.conversion:
            metrics.expected_goals += 0.8  # High probability for actual conversions
        else:
            metrics.expected_goals += 0.1  # Low probability for non-conversions
        
        if FootballRole.ASSISTER.value in roles:
            metrics.expected_assists += 0.7
        elif FootballRole.KEY_PASSER.value in roles:
            metrics.expected_assists += 0.3
    
    def _aggregate_metrics(self, target: FootballMetrics, source: FootballMetrics):
        """Aggregate metrics from source to target."""
        target.goals += source.goals
        target.assists += source.assists
        target.key_passes += source.key_passes
        target.passes += source.passes
        target.minutes += source.minutes
        target.dribbles += source.dribbles
        target.expected_goals += source.expected_goals
        target.expected_assists += source.expected_assists
        target.channel_impact_score += source.channel_impact_score
    
    def _add_football_metrics_to_result(
        self, 
        result_df: pd.DataFrame, 
        channel_metrics: Dict[str, FootballMetrics]
    ) -> pd.DataFrame:
        """Add football metrics columns to the result DataFrame."""
        
        # Create metrics columns
        metrics_columns = [
            'channel_goals', 'channel_assists', 'channel_key_passes',
            'channel_passes', 'channel_minutes', 'channel_dribbles',
            'channel_expected_goals', 'channel_expected_assists',
            'channel_impact_score_total'
        ]
        
        for col in metrics_columns:
            result_df[col] = 0.0
        
        # Populate metrics for each channel
        for channel, metrics in channel_metrics.items():
            channel_mask = result_df['channel'] == channel
            
            result_df.loc[channel_mask, 'channel_goals'] = metrics.goals
            result_df.loc[channel_mask, 'channel_assists'] = metrics.assists
            result_df.loc[channel_mask, 'channel_key_passes'] = metrics.key_passes
            result_df.loc[channel_mask, 'channel_passes'] = metrics.passes
            result_df.loc[channel_mask, 'channel_minutes'] = metrics.minutes
            result_df.loc[channel_mask, 'channel_dribbles'] = metrics.dribbles
            result_df.loc[channel_mask, 'channel_expected_goals'] = metrics.expected_goals
            result_df.loc[channel_mask, 'channel_expected_assists'] = metrics.expected_assists
            result_df.loc[channel_mask, 'channel_impact_score_total'] = metrics.channel_impact_score
        
        return result_df
    
    def _validate_results(self, result_df: pd.DataFrame) -> None:
        """Validate the attribution results."""
        if result_df.empty:
            raise AttributionCalculationError("Attribution results are empty")
        
        # Check that attribution scores are valid
        if 'attribution_score' not in result_df.columns:
            raise AttributionCalculationError("Attribution scores missing from results")
        
        # For each user, check that attribution scores sum to 1.0 (within tolerance)
        tolerance = 1e-6
        
        for user_id in result_df['user_id'].unique():
            user_scores = result_df[result_df['user_id'] == user_id]['attribution_score']
            total_score = user_scores.sum()
            
            if abs(total_score - 1.0) > tolerance:
                raise AttributionCalculationError(
                    f"User {user_id} attribution scores sum to {total_score}, expected 1.0"
                )
        
        logger.debug("Football Attribution results validation passed")
    
    def get_channel_performance_summary(self, result_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get a summary of channel performance with football metrics.
        
        Args:
            result_df: Attribution results DataFrame
            
        Returns:
            DataFrame with channel performance summary
        """
        if result_df.empty:
            return pd.DataFrame()
        
        # Group by channel and aggregate metrics
        summary = result_df.groupby('channel').agg({
            'channel_goals': 'first',
            'channel_assists': 'first', 
            'channel_key_passes': 'first',
            'channel_passes': 'first',
            'channel_minutes': 'first',
            'channel_dribbles': 'first',
            'channel_expected_goals': 'first',
            'channel_expected_assists': 'first',
            'channel_impact_score_total': 'first',
            'attribution_score': 'sum',
            'channel_archetype': 'first'
        }).reset_index()
        
        # Calculate additional metrics
        summary['goals_per_100_passes'] = (summary['channel_goals'] * 100 / summary['channel_passes']).fillna(0)
        summary['assists_per_100_passes'] = (summary['channel_assists'] * 100 / summary['channel_passes']).fillna(0)
        summary['avg_minutes_per_pass'] = (summary['channel_minutes'] / summary['channel_passes']).fillna(0)
        
        # Sort by total attribution score
        summary = summary.sort_values('attribution_score', ascending=False)
        
        return summary
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Football Attribution model."""
        base_info = super().get_model_info()
        
        model_specific_info = {
            'attribution_logic': 'Assigns football roles and calculates Channel Impact Score (CIS)',
            'football_roles': [role.value for role in FootballRole],
            'channel_archetypes': [archetype.value for archetype in ChannelArchetype],
            'use_cases': [
                'Team-based marketing attribution',
                'Channel role identification',
                'Football-inspired analytics',
                'Multi-touch attribution with role weighting'
            ],
            'advantages': [
                'Intuitive football metaphor',
                'Role-based credit assignment',
                'Comprehensive channel metrics',
                'Customizable role weights'
            ],
            'limitations': [
                'Requires engagement time data for best results',
                'Complex role assignment logic',
                'May need role weight tuning'
            ],
            'requires_conversions': True,
            'supports_attribution_window': True,
            'attribution_window_days': self.config.attribution_window_days,
            'role_weights': {
                'scorer': self.football_config.scorer_weight,
                'assister': self.football_config.assister_weight,
                'key_passer': self.football_config.key_passer_weight,
                'most_passes': self.football_config.most_passes_weight,
                'most_minutes': self.football_config.most_minutes_weight,
                'most_dribbles': self.football_config.most_dribbles_weight,
                'participant': self.football_config.participant_weight
            }
        }
        
        base_info.update(model_specific_info)
        return base_info