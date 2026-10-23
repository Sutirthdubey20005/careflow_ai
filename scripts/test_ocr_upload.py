import urllib.request
import json
import io
from PIL import Image, ImageDraw

# Generate lab report image
img = Image.new('RGB', (700, 250), color=(255, 255, 255))
d = ImageDraw.Draw(img)
d.text((10, 15), 'APOLLO CLINICAL DIAGNOSTICS - COMPLETE BLOOD COUNT', fill=(0, 0, 0))
d.text((10, 50), 'Platelet Count: 45000 /mcL (Critical Low)', fill=(0, 0, 0))
d.text((10, 85), 'Total WBC: 18500 /mcL (Severe Infection / Leukocytosis)', fill=(0, 0, 0))
d.text((10, 120), 'Hemoglobin: 8.5 g/dL', fill=(0, 0, 0))
d.text((10, 155), 'Blood Pressure: 160/100 mmHg, Heart Rate: 115 bpm, SpO2: 91%', fill=(0, 0, 0))
buf = io.BytesIO()
img.save(buf, format='JPEG')
img_bytes = buf.getvalue()

# 1. Create case
req1 = urllib.request.Request(
    'http://127.0.0.1:8000/cases',
    data=json.dumps({'chief_complaint': 'Patient uploaded lab report'}).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
case = json.loads(urllib.request.urlopen(req1).read().decode())
case_id = case['id']

# 2. Upload document via multipart
boundary = 'WebKitFormBoundary7MA4YWxkTrZu0gW'
body = (
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="file"; filename="apollo_cbc_report.jpg"\r\n'
    f'Content-Type: image/jpeg\r\n\r\n'
).encode('utf-8') + img_bytes + f'\r\n--{boundary}--\r\n'.encode('utf-8')

req2 = urllib.request.Request(
    f'http://127.0.0.1:8000/cases/{case_id}/upload-document',
    data=body,
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
)
res2 = json.loads(urllib.request.urlopen(req2).read().decode())
print('Upload Result Status:', res2.get('status'))
print('Engine Used:', res2.get('engine_used'))
print('Extracted Text Preview:\n', res2.get('extracted_text')[:300])

# 3. Next Question
req3 = urllib.request.Request(
    f'http://127.0.0.1:8000/cases/{case_id}/next-question',
    data=json.dumps({}).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
q = json.loads(urllib.request.urlopen(req3).read().decode())
print('\nFollow-Up Question Generated from OCR Document!')
print('Next Q EN:', q.get('next_question_en'))
print('Target Field:', q.get('target_field'))
