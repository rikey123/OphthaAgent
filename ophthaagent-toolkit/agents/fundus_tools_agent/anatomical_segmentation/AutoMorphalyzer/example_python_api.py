"""
Example: Using analyze_single.py as a Python module

This demonstrates how to use the analyzer programmatically
instead of calling it from the command line.
"""

import sys
import os
import json

# Add the AutoMorphalyzer directory to the path
sys.path.insert(0, '<ANON_ABS_PATH>')

# Import the analyzer functions
from analyze_single import analyze_images, csv_to_json


def example_1_single_image():
    """Example 1: Analyze a single image"""
    print("="*60)
    print("Example 1: Analyze single image programmatically")
    print("="*60)

    # Define paths
    input_path = 'example_data/test_images/100.png'
    output_dir = 'results/python_api_single'

    # Run analysis
    print(f"\nAnalyzing: {input_path}")
    csv_path = analyze_images(
        input_path=input_path,
        output_dir=output_dir,
        batch_mode=False
    )

    # Convert to JSON
    json_path = os.path.join(output_dir, 'results.json')
    results = csv_to_json(csv_path, json_path)

    # Access the data
    img_data = results['images'][0]
    print(f"\nResults for {img_data['filename']}:")
    print(f"  - Laterality: {img_data['optic_disc']['laterality']}")
    print(f"  - Quality: {img_data['quality']['quickqual_score']:.3f}")
    print(f"  - Vessel Density: {img_data['vessel_features']['binary']['whole']['vessel_density']:.3f}")

    if img_data['large_vessel_metrics']['AVR_B'] != -1:
        print(f"  - AVR (Zone B): {img_data['large_vessel_metrics']['AVR_B']:.3f}")

    return results


def example_2_batch_processing():
    """Example 2: Batch process a directory"""
    print("\n" + "="*60)
    print("Example 2: Batch process directory")
    print("="*60)

    # Define paths
    input_dir = 'example_data/test_images'
    output_dir = 'results/python_api_batch'

    # Run analysis
    print(f"\nAnalyzing directory: {input_dir}")
    csv_path = analyze_images(
        input_path=input_dir,
        output_dir=output_dir,
        batch_mode=True
    )

    # Convert to JSON
    json_path = os.path.join(output_dir, 'batch_results.json')
    results = csv_to_json(csv_path, json_path)

    # Process all results
    print(f"\nProcessed {len(results['images'])} images:")
    for img in results['images']:
        print(f"\n  {img['filename']}:")
        print(f"    - Laterality: {img['optic_disc']['laterality']}")

        avr_b = img['large_vessel_metrics']['AVR_B']
        if avr_b != -1:
            print(f"    - AVR_B: {avr_b:.3f}")

    return results


def example_3_extract_metrics():
    """Example 3: Extract specific metrics from results"""
    print("\n" + "="*60)
    print("Example 3: Extract and analyze metrics")
    print("="*60)

    # Load existing results
    json_path = 'results/python_api_batch/batch_results.json'

    if not os.path.exists(json_path):
        print(f"\nRun example_2_batch_processing() first to generate: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract AVR values
    print("\nArteriovenous Ratio (AVR) Summary:")
    print("-" * 60)
    print(f"{'Filename':<20} {'Laterality':<10} {'AVR_B':<10} {'AVR_C':<10}")
    print("-" * 60)

    for img in data['images']:
        filename = img['filename'][:18]  # Truncate for display
        laterality = img['optic_disc']['laterality']
        avr_b = img['large_vessel_metrics']['AVR_B']
        avr_c = img['large_vessel_metrics']['AVR_C']

        avr_b_str = f"{avr_b:.3f}" if avr_b != -1 else "N/A"
        avr_c_str = f"{avr_c:.3f}" if avr_c != -1 else "N/A"

        print(f"{filename:<20} {laterality:<10} {avr_b_str:<10} {avr_c_str:<10}")

    # Calculate statistics
    valid_avrs = [img['large_vessel_metrics']['AVR_B']
                  for img in data['images']
                  if img['large_vessel_metrics']['AVR_B'] != -1]

    if valid_avrs:
        import statistics
        print("\nStatistics:")
        print(f"  - Mean AVR_B: {statistics.mean(valid_avrs):.3f}")
        print(f"  - Median AVR_B: {statistics.median(valid_avrs):.3f}")
        print(f"  - Std Dev: {statistics.stdev(valid_avrs):.3f}" if len(valid_avrs) > 1 else "")


def example_4_export_to_database():
    """Example 4: Export results to SQLite database"""
    print("\n" + "="*60)
    print("Example 4: Export to SQLite database")
    print("="*60)

    import sqlite3

    # Load results
    json_path = 'results/python_api_batch/batch_results.json'

    if not os.path.exists(json_path):
        print(f"\nRun example_2_batch_processing() first")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create database
    db_path = 'results/retinal_analysis.db'
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Create table
    c.execute('''CREATE TABLE IF NOT EXISTS measurements
                 (filename TEXT PRIMARY KEY,
                  laterality TEXT,
                  quality REAL,
                  avr_b REAL,
                  avr_c REAL,
                  vessel_density REAL,
                  fractal_dimension REAL,
                  disc_height REAL,
                  disc_width REAL,
                  cdr_vertical REAL)''')

    # Insert data
    for img in data['images']:
        c.execute('''INSERT OR REPLACE INTO measurements VALUES (?,?,?,?,?,?,?,?,?,?)''',
                  (img['filename'],
                   img['optic_disc']['laterality'],
                   img['quality']['quickqual_score'],
                   img['large_vessel_metrics']['AVR_B'],
                   img['large_vessel_metrics']['AVR_C'],
                   img['vessel_features']['binary']['whole']['vessel_density'],
                   img['vessel_features']['binary']['whole']['fractal_dimension'],
                   img['optic_disc']['disc_height_px'],
                   img['optic_disc']['disc_width_px'],
                   img['optic_disc']['cdr_vertical']))

    conn.commit()

    # Query and display
    print(f"\nDatabase created: {db_path}")
    print("\nQuery: SELECT filename, laterality, avr_b FROM measurements")
    print("-" * 60)

    for row in c.execute('SELECT filename, laterality, avr_b FROM measurements'):
        print(f"{row[0]:<30} {row[1]:<10} {row[2]:.3f}" if row[2] != -1 else f"{row[0]:<30} {row[1]:<10} N/A")

    conn.close()


def example_5_generate_report():
    """Example 5: Generate HTML report"""
    print("\n" + "="*60)
    print("Example 5: Generate HTML report")
    print("="*60)

    # Load results
    json_path = 'results/python_api_batch/batch_results.json'

    if not os.path.exists(json_path):
        print(f"\nRun example_2_batch_processing() first")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Generate HTML
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Retinal Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #2c3e50; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #3498db; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .metric { font-weight: bold; color: #2c3e50; }
    </style>
</head>
<body>
    <h1>Retinal Vessel Analysis Report</h1>
    <p><strong>Generated:</strong> {timestamp}</p>
    <p><strong>Images Analyzed:</strong> {num_images}</p>

    <h2>Results Summary</h2>
    <table>
        <tr>
            <th>Filename</th>
            <th>Laterality</th>
            <th>Quality</th>
            <th>AVR (B)</th>
            <th>Vessel Density</th>
            <th>Fractal Dimension</th>
        </tr>
""".format(
        timestamp=data['metadata']['timestamp'],
        num_images=len(data['images'])
    )

    # Add rows
    for img in data['images']:
        avr_b = img['large_vessel_metrics']['AVR_B']
        avr_str = f"{avr_b:.3f}" if avr_b != -1 else "N/A"

        html += f"""
        <tr>
            <td>{img['filename']}</td>
            <td>{img['optic_disc']['laterality']}</td>
            <td>{img['quality']['quickqual_score']:.3f}</td>
            <td>{avr_str}</td>
            <td>{img['vessel_features']['binary']['whole']['vessel_density']:.3f}</td>
            <td>{img['vessel_features']['binary']['whole']['fractal_dimension']:.3f}</td>
        </tr>
"""

    html += """
    </table>
</body>
</html>
"""

    # Save report
    report_path = 'results/analysis_report.html'
    with open(report_path, 'w') as f:
        f.write(html)

    print(f"\nHTML report generated: {report_path}")
    print("Open it in a web browser to view the results")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("  AutoMorphalyzer Python API Examples")
    print("="*60)

    # Check if we're in the right directory
    if not os.path.exists('automorph'):
        print("\nERROR: Please run this script from the AutoMorphalyzer root directory")
        return

    # Run examples
    try:
        # Example 1: Single image
        example_1_single_image()

        # Example 2: Batch processing
        example_2_batch_processing()

        # Example 3: Extract metrics
        example_3_extract_metrics()

        # Example 4: Database export
        example_4_export_to_database()

        # Example 5: HTML report
        example_5_generate_report()

        print("\n" + "="*60)
        print("✓ All examples completed successfully!")
        print("="*60)
        print("\nGenerated files:")
        print("  - results/python_api_single/results.json")
        print("  - results/python_api_batch/batch_results.json")
        print("  - results/retinal_analysis.db")
        print("  - results/analysis_report.html")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
