#!/usr/bin/env python3
"""Validation script for the universal Overture feature retriever implementation.

This script should be run inside the dev container with the 'gis' conda environment
activated to properly test the implementation.

Usage:
    python validate_implementation.py [data_path]
"""

import sys
import traceback
from pathlib import Path


def validate_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import duckdb
        print("✅ DuckDB imported successfully")
    except ImportError as e:
        print(f"❌ DuckDB import failed: {e}")
        return False
    
    try:
        from pyarrow import parquet as pq
        print("✅ PyArrow imported successfully")
    except ImportError as e:
        print(f"❌ PyArrow import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    return True


def validate_spatial_extension():
    """Test that DuckDB spatial extension can be loaded."""
    print("\n🔍 Testing DuckDB spatial extension...")
    
    try:
        import duckdb
        conn = duckdb.connect(":memory:")
        
        # Try to install and load spatial extension
        try:
            conn.execute("LOAD spatial;")
            print("✅ Spatial extension already available")
        except:
            print("📦 Installing spatial extension...")
            conn.execute("INSTALL spatial;")
            conn.execute("LOAD spatial;")
            print("✅ Spatial extension installed and loaded")
        
        # Test basic spatial function
        result = conn.execute("SELECT ST_Point(0, 0) as geom;").fetchone()
        print("✅ Basic spatial operations working")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Spatial extension test failed: {e}")
        traceback.print_exc()
        return False


def validate_retriever_classes():
    """Test that our retriever classes can be imported and instantiated."""
    print("\n🔍 Testing retriever class imports...")
    
    try:
        # Import our classes
        from overture_feature_retriever import (
            OvertureFeatureRetriever,
            MultiFeatureRetriever,
            FeatureTypeInfo
        )
        print("✅ Retriever classes imported successfully")
        
        # Test static methods
        themes = OvertureFeatureRetriever.get_available_themes("/nonexistent")
        assert themes == []
        print("✅ Static method get_available_themes works")
        
        types = OvertureFeatureRetriever.get_available_types("/nonexistent", "buildings")
        assert types == []
        print("✅ Static method get_available_types works")
        
        # Test MultiFeatureRetriever initialization
        multi = MultiFeatureRetriever("/nonexistent")
        assert multi.get_feature_types() == []
        print("✅ MultiFeatureRetriever initialization works")
        
        return True
        
    except Exception as e:
        print(f"❌ Retriever class test failed: {e}")
        traceback.print_exc()
        return False


def validate_with_real_data(data_path):
    """Test with real Overture data if available."""
    print(f"\n🔍 Testing with real data at {data_path}...")
    
    data_path = Path(data_path)
    if not data_path.exists():
        print(f"⚠️  Data path {data_path} does not exist, skipping real data test")
        return True
    
    try:
        from overture_feature_retriever import OvertureFeatureRetriever
        
        # Discover available themes
        themes = OvertureFeatureRetriever.get_available_themes(data_path)
        print(f"📊 Found themes: {themes}")
        
        if not themes:
            print("⚠️  No themes found, skipping real data test")
            return True
        
        # Test with first available theme
        theme = themes[0]
        types = OvertureFeatureRetriever.get_available_types(data_path, theme)
        print(f"📊 Found types for {theme}: {types}")
        
        if not types:
            print(f"⚠️  No types found for {theme}, skipping real data test")
            return True
        
        # Try to initialize retriever with first available type
        feature_type = types[0]
        print(f"🔧 Testing retriever for {theme}/{feature_type}...")
        
        retriever = OvertureFeatureRetriever(data_path, theme, feature_type)
        print(f"✅ Retriever initialized successfully")
        print(f"   - Geometry column: {retriever.geometry_column}")
        print(f"   - File count: {retriever.file_count}")
        print(f"   - Schema info keys: {list(retriever.schema_info.keys())}")
        
        # Test a small bbox query
        print("🔧 Testing small bbox query...")
        result = retriever.query_bbox(-1, -1, 1, 1, limit=5)
        print(f"✅ Query successful, returned {len(result)} rows")
        if not result.empty:
            print(f"   - Columns: {list(result.columns)}")
        
        retriever.close()
        return True
        
    except Exception as e:
        print(f"❌ Real data test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("🚀 Validating Overture Feature Retriever Implementation")
    print("=" * 60)
    
    # Get data path from command line or use default
    data_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/gis_data"
    
    all_passed = True
    
    # Run validation tests
    all_passed &= validate_imports()
    all_passed &= validate_spatial_extension()
    all_passed &= validate_retriever_classes()
    all_passed &= validate_with_real_data(data_path)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All validation tests passed!")
        print("\nThe universal feature retriever infrastructure is ready to use.")
        print("\nNext steps:")
        print("1. Use OvertureFeatureRetriever for single feature type analysis")
        print("2. Use MultiFeatureRetriever for comparative analysis")
        print("3. Proceed to implement task 2: source analysis functionality")
    else:
        print("❌ Some validation tests failed!")
        print("\nPlease check the errors above and ensure you're running in the")
        print("dev container with the 'gis' conda environment activated.")
        sys.exit(1)


if __name__ == "__main__":
    main()