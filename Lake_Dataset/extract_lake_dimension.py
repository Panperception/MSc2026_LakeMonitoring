import ee, os

# User Input
country_name = "Canada"
base_dir = f"/content/drive/MyDrive/{country_name}_Lakes"

# Initialize Earth Engine
ee.Authenticate()
ee.Initialize(project="lake-study-2026")

# Create Lake directory
os.makedirs(base_dir, exist_ok=True)

# Function to Assign lake_id property to zipped items of each lake with associated sequence
def assign_lakeid(zip_item):
    zip_item = ee.List(zip_item)

    # Extract lake features and sequence numbers from zip item
    lake = ee.Feature(zip_item.get(0))
    sequence = ee.Number(zip_item.get(1))

    # Generate sequential ID
    seq_id = ee.String("lake_").cat(sequence.format("%d"))

    # Save sequential ID as new lake_id property
    return lake.set('Lake_id', seq_id)

# Function to Extract rectangle dimensions for each lake using a server-side map function with parallel processing
def extract_dimensions(lake, buffer=0):
    # Add buffer around the geometry (buffer value in meters)
    # Get the bounding box coordinates list
    raw_bbox = lake.geometry().buffer(buffer).bounds().coordinates().get(0)
    bbox = ee.List(raw_bbox)

    # Extract the min and max corners from the 5-point coordinates list
    p0 = ee.List(bbox.get(0)) # Bottom-left / Sounth-west corner [min_lng, min_lat]
    p2 = ee.List(bbox.get(2)) # Top-right / North-east corner [max_lng, max_lat]

    # Store dimensions as lake's new properties
    return lake.set({
        'min_lng': p0.get(0),
        'min_lat': p0.get(1),
        'max_lng': p2.get(0),
        'max_lat': p2.get(1)
    })

# Load the global HydroLAKES asset
hydro_lakes = ee.FeatureCollection("projects/lake-study-2026/assets/HydroLAKES")
print("Global HydroLAKES count:",hydro_lakes.size().getInfo())

# Get country region geometry
# Load the USDOS 2017 LSIB (Large Scale International Boundaries) dataset
#countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
#region = countries.filter(ee.Filter.eq("country_na", country_name)).first()
#region_polygon = region.geometry()

# Filter lakes inside country's boundary
#region_lakes = hydro_lakes.filterBounds(region_polygon)

# Filter natural lakes only (Lake type = 1) with lake area >= 5km2
natural_lakes = hydro_lakes.filter(ee.Filter.eq('Lake_type', 1)) \
                .filter(ee.Filter.gte('Lake_area', 5.0)) \
                .filter(ee.Filter.eq('Country', country_name)) \
                .sort('Lake_area', False);

# Extract lakes with valid registered names
named_lakes = natural_lakes \
    .filter(ee.Filter.notNull(["Lake_name"])) \
    .filter(ee.Filter.neq("Lake_name", ""))

# Generate sequential numbers list and combine with filtered lakes
# Setup lake_id property for lakes sequential naming
lake_list = ee.List(ee.Algorithms.If(
                named_lakes.size().gte(100),
                named_lakes.toList(100),
                natural_lakes.toList(100)))
seq_list = ee.List.sequence(1, lake_list.size())
zip_list = lake_list.zip(seq_list)
updated_lakes = ee.FeatureCollection(zip_list.map(assign_lakeid))

# Apply the dimension extraction across all filtered lakes with 10km (10000m) buffer
#candidate_lakes = updated_lakes.map(extract_dimensions)
candidate_lakes = updated_lakes.map(lambda lake: extract_dimensions(lake, buffer=10000))

# Fetch data from Earth Engine servers using .getInfo()
try:
    #total_lake_count = region_lakes.size().getInfo()
    natural_lake_count = natural_lakes.size().getInfo()
    named_lake_count = named_lakes.size().getInfo()
    lake_count = candidate_lakes.size().getInfo()

    # Print information
    print("\nCountry:",country_name)
    #print(f"\nTotal lakes found in {country_name}: {total_lake_count}")
    print(f"Natural lakes with area > 5km2 found: {natural_lake_count}")
    print(f"Natural lakes with valid names: {named_lake_count}")
    print(f"Candidate lakes for analysis: {lake_count}")

    # Fetch sample lakes with updated metadata
    sample_features = candidate_lakes.limit(100).getInfo()['features']
    #sample_features = candidate_lakes.filter(ee.Filter.eq('Lake_name', 'Tshchikskoye')).getInfo()['features']

    lake_info = []
    #lake_dict = {}
    print(f"\nSample list of candidate lakes metadata:")
    print(f"  ID : Name -> Rectangle Coordintaes -> Area")
    for feature in sample_features:
        props = feature['properties']
        lake_name = props.get('Lake_name')
        lake_id = props.get('Lake_id')
        lake_area = props.get('Lake_area')
        lake_coords = [props.get('min_lng'), props.get('min_lat'), props.get('max_lng'), props.get('max_lat')]
        lake_coords = [round(c, 2) if isinstance(c, (int, float)) else c for c in lake_coords]   # Round-off to 2 decimal points
        print(f"• {lake_id} : {lake_name} -> {lake_coords} -> {lake_area}")
        lake_info.append((lake_id, lake_coords))
        #lake_dict[lake_id] = lake_name
    print(f"\nLake Info List :\n{lake_info}")
except Exception as e:
    print(f"An error occurred while fetching data: {e}")