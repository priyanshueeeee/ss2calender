import google.generativeai as genai
from PIL import Image
import json

GEMINI_API_KEY = "API_KEY"
image_path = "timetable.jpg"

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

try:
    image = Image.open(image_path)
except FileNotFoundError:
    print(f"❌ Image file '{image_path}' not found!")
    print("Please make sure the image file exists in the current directory.")
    exit(1)

prompt = """
This is an academic timetable image showing a weekly schedule from Monday to Friday.

IMPORTANT: The timetable uses these EXACT time slots:
- 8:00-8:50 AM
- 9:00-9:50 AM  
- 10:00-10:50 AM
- 11:00-11:50 AM
- 12:00-12:50 PM
- 2:00-2:50 PM
- 3:00-3:50 PM
- 4:00-4:50 PM
- 5:00-5:50 PM
- 6:00-6:50 PM
- 7:00-7:50 PM

STEP-BY-STEP EXTRACTION METHOD:
1. Look at the image and identify the 5 day columns: Monday (leftmost), Tuesday, Wednesday, Thursday, Friday (rightmost)
2. For each day column, scan vertically from top to bottom (8:00AM to 7:00PM)
3. When you find a green box in a day column, extract its information
4. Each green box = one class session
5. Consecutive green boxes are DIFFERENT classes (even if they appear adjacent)
6. Some classes may span multiple time slots (labs), but consecutive boxes with different text are separate

CRITICAL DAY ASSIGNMENT RULES:
- ONLY extract classes from the specific day column you're currently processing
- NEVER copy a class from one day column to another day
- Each class must be assigned to the day column where it actually appears
- A course can have multiple lectures/tutorials/labs in different days of the week (this is normal)
- If you see the same course code in multiple days, this is likely correct - verify by checking the image
- The same course should NOT appear in the same time slot on multiple days (this would be an error)

EXTRACTION ORDER:
1. Process Monday column (leftmost) - extract ALL green boxes in this column only
2. Process Tuesday column - extract ALL green boxes in this column only  
3. Process Wednesday column - extract ALL green boxes in this column only
4. Process Thursday column - extract ALL green boxes in this column only
5. Process Friday column (rightmost) - extract ALL green boxes in this column only

For each class session, extract:
- day (Monday, Tuesday, Wednesday, Thursday, Friday)
- start_time (in 12-hour format like "9:00AM", "2:00PM")
- end_time (in 12-hour format like "9:50AM", "2:50PM")
- course_code (e.g., "CS F213", "ECE F241")
- course_name (full course name)
- class_type (Lecture, Tutorial, Laboratory/Lab)
- location (room/venue)
- instructor (name or "Staff" if not specified)

FINAL CHECK:
- Verify each class is in the correct day column
- Ensure no class appears in multiple days unless actually scheduled that way
- Count classes per day to verify accuracy

Only return the JSON. Do not add explanation or comments.
"""

# === STEP 4: Make Request to Gemini ===
response = model.generate_content(
    [prompt, image],
    stream=False,
)

# === STEP 5: Get and Save Output ===
try:
    output = response.text.strip()
    
    if output.startswith('```json'):
        output = output[7:] 
    if output.endswith('```'):
        output = output[:-3] 
    
    output = output.strip()
    
    json_data = json.loads(output)
    
    with open("timetable.json", "w") as f:
        json.dump(json_data, f, indent=2)

    print("✅ Timetable saved to timetable.json")
    print(f"📊 Extracted {len(json_data)} class sessions")
    
except json.JSONDecodeError as e:
    print("❌ Failed to parse response as JSON:")
    print("Raw response:")
    print(response.text)
    print(f"\nJSON Error: {e}")
    print(f"Error position: line {e.lineno}, column {e.colno}")
    
    with open("raw_response.txt", "w") as f:
        f.write(response.text)
    print("💾 Raw response saved to raw_response.txt for debugging")
    
except Exception as e:
    print("❌ Unexpected error:")
    print(f"Error: {e}")
    print("Raw response:")
    print(response.text)
