"""Source analysis and extraction functionality for Overture Maps data.

This module provides comprehensive analysis of data sources within Overture Maps features,
extracting and categorizing source attribution information from nested structures across
different feature types.

Key Features:
- Source extraction from complex nested structures in the 'sources' field
- Statistical analysis of source distribution patterns
- Data quality assessment based on confidence scores and metadata
- Comparative analysis across regions and feature types
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union, Tuple

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceInfo:
    """Information about a data source extracted from Overture features."""
    dataset: str
    property: Optional[str] = None
    record_id: Optional[str] = None
    confidence: Optional[float] = None
    update_time: Optional[str] = None


@dataclass(frozen=True)
class SourceStatistics:
    """Statistical summary of source analysis results."""
    total_features: int
    total_sources: int
    unique_datasets: int
    source_breakdown: Dict[str, int]
    source_percentages: Dict[str, float]
    confidence_stats: Dict[str, float]
    coverage_metrics: Dict[str, float]


class SourceAnalyzer:
    """Analyzes data source composition and patterns in Overture Maps features.
    
    This class extracts source attribution information from the 'sources' field
    of Overture features and provides statistical analysis capabilities.
    
    Parameters
    ----------
    source_column : str, default 'sources'
        Name of the column containing source information
    confidence_threshold : float, default 0.0
        Minimum confidence score to include in analysis
    """
    
    def __init__(
        self,
        source_column: str = "sources",
        confidence_threshold: float = 0.0
    ) -> None:
        self.source_column = source_column
        self.confidence_threshold = confidence_threshold
    
    def extract_sources(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Extract and normalize source information from features DataFrame.
        
        Processes the nested 'sources' field to extract individual source records
        with their metadata including dataset, property, record_id, confidence, etc.
        
        Parameters
        ----------
        features_df : pd.DataFrame
            DataFrame containing Overture features with source information
            
        Returns
        -------
        pd.DataFrame
            Normalized DataFrame with one row per source record, containing:
            - feature_id: Original feature identifier
            - dataset: Source dataset name
            - property: Source property (if available)
            - record_id: Source record identifier (if available)
            - confidence: Confidence score (if available)
            - update_time: Last update timestamp (if available)
        """
        if self.source_column not in features_df.columns:
            logger.warning(f"Source column '{self.source_column}' not found in DataFrame")
            return pd.DataFrame(columns=[
                'feature_id', 'dataset', 'property', 'record_id', 'confidence', 'update_time'
            ])
        
        sources_data = []
        
        for idx, row in features_df.iterrows():
            feature_id = row.get('id', idx)  # Use 'id' column if available, otherwise row index
            sources = row[self.source_column]
            
            if pd.isna(sources) or sources is None:
                continue
            
            # Handle different source data formats
            parsed_sources = self._parse_sources_field(sources)
            
            for source_info in parsed_sources:
                # Apply confidence threshold filter
                if (source_info.confidence is not None and 
                    source_info.confidence < self.confidence_threshold):
                    continue
                
                sources_data.append({
                    'feature_id': feature_id,
                    'dataset': source_info.dataset,
                    'property': source_info.property,
                    'record_id': source_info.record_id,
                    'confidence': source_info.confidence,
                    'update_time': source_info.update_time
                })
        
        return pd.DataFrame(sources_data)
    
    def _parse_sources_field(self, sources: Any) -> List[SourceInfo]:
        """Parse the sources field which can be in various formats.
        
        Parameters
        ----------
        sources : Any
            Sources field value (could be string, list, dict, etc.)
            
        Returns
        -------
        List[SourceInfo]
            List of parsed source information objects
        """
        if sources is None or pd.isna(sources):
            return []
        
        try:
            # If it's a string, try to parse as JSON
            if isinstance(sources, str):
                if sources.strip().startswith('[') or sources.strip().startswith('{'):
                    sources = json.loads(sources)
                else:
                    # Simple string dataset name
                    return [SourceInfo(dataset=sources)]
            
            # Handle list of sources
            if isinstance(sources, list):
                source_infos = []
                for source in sources:
                    source_infos.extend(self._parse_single_source(source))
                return source_infos
            
            # Handle single source object
            elif isinstance(sources, dict):
                return self._parse_single_source(sources)
            
            # Handle other types by converting to string
            else:
                return [SourceInfo(dataset=str(sources))]
                
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.debug(f"Failed to parse sources field: {sources}, error: {e}")
            # Fallback: treat as simple string dataset name
            return [SourceInfo(dataset=str(sources))]
    
    def _parse_single_source(self, source: Any) -> List[SourceInfo]:
        """Parse a single source object.
        
        Parameters
        ----------
        source : Any
            Single source object (dict, string, etc.)
            
        Returns
        -------
        List[SourceInfo]
            List containing the parsed source info (usually single item)
        """
        if isinstance(source, dict):
            # Extract fields with various possible key names
            dataset = (source.get('dataset') or 
                      source.get('source') or 
                      source.get('name') or 
                      source.get('provider') or
                      'unknown')
            
            property_name = source.get('property')
            record_id = source.get('record_id') or source.get('recordId') or source.get('id')
            
            # Handle confidence score
            confidence = source.get('confidence')
            if confidence is not None:
                try:
                    confidence = float(confidence)
                except (ValueError, TypeError):
                    confidence = None
            
            update_time = source.get('update_time') or source.get('updateTime')
            
            return [SourceInfo(
                dataset=str(dataset),
                property=property_name,
                record_id=record_id,
                confidence=confidence,
                update_time=update_time
            )]
        
        elif isinstance(source, str):
            return [SourceInfo(dataset=source)]
        
        else:
            return [SourceInfo(dataset=str(source))]
    
    def categorize_sources(self, sources_df: pd.DataFrame) -> Dict[str, int]:
        """Categorize sources by dataset and return counts.
        
        Parameters
        ----------
        sources_df : pd.DataFrame
            DataFrame from extract_sources() containing source records
            
        Returns
        -------
        Dict[str, int]
            Dictionary mapping dataset names to feature counts
        """
        if sources_df.empty:
            return {}
        
        # Count unique features per dataset (not total source records)
        dataset_counts = (sources_df.groupby('dataset')['feature_id']
                         .nunique()
                         .to_dict())
        
        return dict(sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True))
    
    def calculate_coverage_metrics(self, sources_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate coverage and quality metrics for the source data.
        
        Parameters
        ----------
        sources_df : pd.DataFrame
            DataFrame from extract_sources() containing source records
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing various coverage metrics:
            - features_with_sources: Percentage of features that have source information
            - avg_sources_per_feature: Average number of sources per feature
            - confidence_coverage: Percentage of sources with confidence scores
            - avg_confidence: Average confidence score (where available)
            - dataset_diversity: Shannon diversity index of dataset distribution
        """
        if sources_df.empty:
            return {
                'features_with_sources': 0.0,
                'avg_sources_per_feature': 0.0,
                'confidence_coverage': 0.0,
                'avg_confidence': 0.0,
                'dataset_diversity': 0.0
            }
        
        # Basic coverage metrics
        unique_features = sources_df['feature_id'].nunique()
        total_source_records = len(sources_df)
        
        # Confidence metrics
        confidence_records = sources_df.dropna(subset=['confidence'])
        confidence_coverage = len(confidence_records) / total_source_records if total_source_records > 0 else 0.0
        avg_confidence = confidence_records['confidence'].mean() if not confidence_records.empty else 0.0
        
        # Dataset diversity (Shannon entropy)
        dataset_counts = sources_df.groupby('dataset')['feature_id'].nunique()
        total_features = dataset_counts.sum()
        if total_features > 0:
            proportions = dataset_counts / total_features
            # Shannon diversity index
            dataset_diversity = -sum(p * np.log(p) for p in proportions if p > 0)
        else:
            dataset_diversity = 0.0
        
        return {
            'features_with_sources': 100.0,  # All features in sources_df have sources by definition
            'avg_sources_per_feature': total_source_records / unique_features if unique_features > 0 else 0.0,
            'confidence_coverage': confidence_coverage * 100.0,
            'avg_confidence': float(avg_confidence) if not pd.isna(avg_confidence) else 0.0,
            'dataset_diversity': float(dataset_diversity)
        }
    
    def assess_data_quality(self, sources_df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality based on source metadata completeness and consistency.
        
        Parameters
        ----------
        sources_df : pd.DataFrame
            DataFrame from extract_sources() containing source records
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing data quality assessment:
            - completeness_scores: Percentage of records with each field populated
            - confidence_distribution: Statistics about confidence score distribution
            - dataset_consistency: Metrics about dataset naming consistency
            - temporal_coverage: Information about update time distribution
        """
        if sources_df.empty:
            return {
                'completeness_scores': {},
                'confidence_distribution': {},
                'dataset_consistency': {},
                'temporal_coverage': {}
            }
        
        total_records = len(sources_df)
        
        # Completeness scores
        completeness_scores = {}
        for column in ['dataset', 'property', 'record_id', 'confidence', 'update_time']:
            non_null_count = sources_df[column].notna().sum()
            completeness_scores[column] = (non_null_count / total_records) * 100.0
        
        # Confidence distribution
        confidence_data = sources_df.dropna(subset=['confidence'])
        if not confidence_data.empty:
            confidence_distribution = {
                'count': len(confidence_data),
                'mean': float(confidence_data['confidence'].mean()),
                'std': float(confidence_data['confidence'].std()),
                'min': float(confidence_data['confidence'].min()),
                'max': float(confidence_data['confidence'].max()),
                'q25': float(confidence_data['confidence'].quantile(0.25)),
                'q50': float(confidence_data['confidence'].quantile(0.50)),
                'q75': float(confidence_data['confidence'].quantile(0.75))
            }
        else:
            confidence_distribution = {'count': 0}
        
        # Dataset consistency
        dataset_names = sources_df['dataset'].dropna().unique()
        dataset_consistency = {
            'unique_datasets': len(dataset_names),
            'most_common_dataset': sources_df['dataset'].mode().iloc[0] if not sources_df['dataset'].empty else None,
            'dataset_name_lengths': {
                'min': min(len(name) for name in dataset_names) if dataset_names.size > 0 else 0,
                'max': max(len(name) for name in dataset_names) if dataset_names.size > 0 else 0,
                'avg': np.mean([len(name) for name in dataset_names]) if dataset_names.size > 0 else 0
            }
        }
        
        # Temporal coverage
        temporal_data = sources_df.dropna(subset=['update_time'])
        temporal_coverage = {
            'records_with_timestamps': len(temporal_data),
            'coverage_percentage': (len(temporal_data) / total_records) * 100.0 if total_records > 0 else 0.0
        }
        
        if not temporal_data.empty:
            # Try to parse timestamps for additional analysis
            try:
                temporal_data_parsed = pd.to_datetime(temporal_data['update_time'], errors='coerce')
                valid_timestamps = temporal_data_parsed.dropna()
                if not valid_timestamps.empty:
                    temporal_coverage.update({
                        'earliest_update': str(valid_timestamps.min()),
                        'latest_update': str(valid_timestamps.max()),
                        'valid_timestamps': len(valid_timestamps)
                    })
            except Exception as e:
                logger.debug(f"Failed to parse timestamps: {e}")
        
        return {
            'completeness_scores': completeness_scores,
            'confidence_distribution': confidence_distribution,
            'dataset_consistency': dataset_consistency,
            'temporal_coverage': temporal_coverage
        }
    
    def create_summary_statistics(
        self, 
        features_df: pd.DataFrame,
        region_id: Optional[str] = None
    ) -> SourceStatistics:
        """Create comprehensive summary statistics for source analysis.
        
        Parameters
        ----------
        features_df : pd.DataFrame
            DataFrame containing Overture features with source information
        region_id : str, optional
            Identifier for the analyzed region
            
        Returns
        -------
        SourceStatistics
            Comprehensive statistics about the source analysis
        """
        sources_df = self.extract_sources(features_df)
        
        if sources_df.empty:
            return SourceStatistics(
                total_features=len(features_df),
                total_sources=0,
                unique_datasets=0,
                source_breakdown={},
                source_percentages={},
                confidence_stats={},
                coverage_metrics={}
            )
        
        # Basic counts
        total_features = len(features_df)
        total_sources = len(sources_df)
        unique_datasets = sources_df['dataset'].nunique()
        
        # Source breakdown and percentages
        source_breakdown = self.categorize_sources(sources_df)
        total_features_with_sources = sum(source_breakdown.values())
        source_percentages = {
            dataset: (count / total_features_with_sources) * 100.0
            for dataset, count in source_breakdown.items()
        } if total_features_with_sources > 0 else {}
        
        # Confidence statistics
        confidence_data = sources_df.dropna(subset=['confidence'])
        if not confidence_data.empty:
            confidence_stats = {
                'mean': float(confidence_data['confidence'].mean()),
                'std': float(confidence_data['confidence'].std()),
                'min': float(confidence_data['confidence'].min()),
                'max': float(confidence_data['confidence'].max()),
                'coverage': (len(confidence_data) / total_sources) * 100.0
            }
        else:
            confidence_stats = {'coverage': 0.0}
        
        # Coverage metrics
        coverage_metrics = self.calculate_coverage_metrics(sources_df)
        
        return SourceStatistics(
            total_features=total_features,
            total_sources=total_sources,
            unique_datasets=unique_datasets,
            source_breakdown=source_breakdown,
            source_percentages=source_percentages,
            confidence_stats=confidence_stats,
            coverage_metrics=coverage_metrics
        )


class ComparativeSourceAnalyzer:
    """Compares source composition across regions or feature types.
    
    This class provides methods for comparative analysis of source distributions,
    identifying coverage gaps, and statistical comparison across different dimensions.
    """
    
    def __init__(self, base_analyzer: Optional[SourceAnalyzer] = None) -> None:
        """Initialize comparative analyzer.
        
        Parameters
        ----------
        base_analyzer : SourceAnalyzer, optional
            Base analyzer to use for source extraction. If None, creates default.
        """
        self.base_analyzer = base_analyzer or SourceAnalyzer()
    
    def compare_regions(
        self, 
        region_analyses: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Compare source composition across multiple regions.
        
        Parameters
        ----------
        region_analyses : List[Dict[str, Any]]
            List of analysis results, each containing:
            - region_id: Region identifier
            - features_df: DataFrame of features for the region
            - Any additional metadata
            
        Returns
        -------
        pd.DataFrame
            Comparison DataFrame with regions as rows and datasets as columns,
            showing feature counts or percentages for each dataset per region
        """
        if not region_analyses:
            return pd.DataFrame()
        
        comparison_data = []
        
        for analysis in region_analyses:
            region_id = analysis.get('region_id', 'unknown')
            features_df = analysis.get('features_df')
            
            if features_df is None or features_df.empty:
                continue
            
            # Get source statistics for this region
            stats = self.base_analyzer.create_summary_statistics(features_df, region_id)
            
            # Create row for comparison
            row_data = {
                'region_id': region_id,
                'total_features': stats.total_features,
                'total_sources': stats.total_sources,
                'unique_datasets': stats.unique_datasets,
                **stats.source_breakdown  # Add dataset counts
            }
            
            comparison_data.append(row_data)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Fill NaN values with 0 for datasets not present in all regions
        dataset_columns = [col for col in comparison_df.columns 
                          if col not in ['region_id', 'total_features', 'total_sources', 'unique_datasets']]
        comparison_df[dataset_columns] = comparison_df[dataset_columns].fillna(0)
        
        return comparison_df
    
    def compare_feature_types(
        self, 
        type_analyses: Dict[Tuple[str, str], Dict[str, Any]]
    ) -> pd.DataFrame:
        """Compare source composition across different feature types.
        
        Parameters
        ----------
        type_analyses : Dict[Tuple[str, str], Dict[str, Any]]
            Dictionary mapping (theme, feature_type) tuples to analysis data:
            - features_df: DataFrame of features for the type
            - Any additional metadata
            
        Returns
        -------
        pd.DataFrame
            Comparison DataFrame with feature types as rows and datasets as columns
        """
        if not type_analyses:
            return pd.DataFrame()
        
        comparison_data = []
        
        for (theme, feature_type), analysis in type_analyses.items():
            features_df = analysis.get('features_df')
            
            if features_df is None or features_df.empty:
                continue
            
            # Get source statistics for this feature type
            type_id = f"{theme}/{feature_type}"
            stats = self.base_analyzer.create_summary_statistics(features_df, type_id)
            
            # Create row for comparison
            row_data = {
                'theme': theme,
                'feature_type': feature_type,
                'type_id': type_id,
                'total_features': stats.total_features,
                'total_sources': stats.total_sources,
                'unique_datasets': stats.unique_datasets,
                **stats.source_breakdown  # Add dataset counts
            }
            
            comparison_data.append(row_data)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Fill NaN values with 0 for datasets not present in all feature types
        dataset_columns = [col for col in comparison_df.columns 
                          if col not in ['theme', 'feature_type', 'type_id', 
                                       'total_features', 'total_sources', 'unique_datasets']]
        comparison_df[dataset_columns] = comparison_df[dataset_columns].fillna(0)
        
        return comparison_df
    
    def identify_coverage_gaps(
        self, 
        analyses: List[Dict[str, Any]],
        min_coverage_threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Identify coverage gaps across analyses.
        
        Parameters
        ----------
        analyses : List[Dict[str, Any]]
            List of analysis results to compare
        min_coverage_threshold : float, default 0.1
            Minimum coverage percentage to not be considered a gap
            
        Returns
        -------
        List[Dict[str, Any]]
            List of identified coverage gaps with details
        """
        if not analyses:
            return []
        
        gaps = []
        
        # Collect all datasets across analyses
        all_datasets = set()
        analysis_stats = []
        
        for analysis in analyses:
            features_df = analysis.get('features_df')
            if features_df is None or features_df.empty:
                continue
            
            region_id = analysis.get('region_id', 'unknown')
            stats = self.base_analyzer.create_summary_statistics(features_df, region_id)
            analysis_stats.append((region_id, stats))
            all_datasets.update(stats.source_breakdown.keys())
        
        # Check for gaps in each analysis
        for region_id, stats in analysis_stats:
            total_features = stats.total_features
            
            for dataset in all_datasets:
                dataset_count = stats.source_breakdown.get(dataset, 0)
                coverage_pct = (dataset_count / total_features) * 100.0 if total_features > 0 else 0.0
                
                if coverage_pct < min_coverage_threshold:
                    gaps.append({
                        'region_id': region_id,
                        'dataset': dataset,
                        'coverage_percentage': coverage_pct,
                        'feature_count': dataset_count,
                        'total_features': total_features,
                        'gap_type': 'low_coverage' if coverage_pct > 0 else 'missing_dataset'
                    })
        
        return sorted(gaps, key=lambda x: x['coverage_percentage'])


__all__ = [
    "SourceAnalyzer",
    "ComparativeSourceAnalyzer", 
    "SourceInfo",
    "SourceStatistics"
]