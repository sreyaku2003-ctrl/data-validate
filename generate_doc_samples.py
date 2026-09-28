"""
Generate sample PDF certificates & Aadhaar documents for testing.
"""

from pathlib import Path
import fitz

SAMPLES_DIR = Path("samples")
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# 1. Sample Aadhaar Card PDF
doc_aadhaar = fitz.open()
p1 = doc_aadhaar.new_page(width=400, height=260)
p1.draw_rect(fitz.Rect(10, 10, 390, 250), color=(0.8, 0.2, 0.2), width=2)
p1.insert_text((30, 40), "Government of India / Bharat Sarkar", fontsize=12, fontname="helv", color=(0.7, 0.1, 0.1))
p1.insert_text((30, 60), "Unique Identification Authority of India (UIDAI)", fontsize=11, fontname="helv")
p1.insert_text((30, 95), "Enrollment No: 1024/50321/98765", fontsize=9, fontname="helv")
p1.insert_text((30, 120), "Name: Aarav Sharma", fontsize=11, fontname="helv")
p1.insert_text((30, 140), "DOB: 15/08/2008 | Gender: Male", fontsize=10, fontname="helv")
p1.insert_text((30, 180), "Aadhaar No: 4321 8765 9876", fontsize=16, fontname="helv", color=(0.1, 0.1, 0.5))
p1.insert_text((30, 215), "Mera Aadhaar, Meri Pehchan", fontsize=10, fontname="helv", color=(0.2, 0.5, 0.2))
doc_aadhaar.save(SAMPLES_DIR / "sample_aadhaar.pdf")

# 2. Sample Birth Certificate PDF
doc_birth = fitz.open()
p2 = doc_birth.new_page(width=450, height=320)
p2.draw_rect(fitz.Rect(15, 15, 435, 305), color=(0.2, 0.4, 0.6), width=2)
p2.insert_text((130, 45), "GOVERNMENT OF KARNATAKA", fontsize=12, fontname="helv")
p2.insert_text((110, 70), "DEPARTMENT OF HEALTH & FAMILY WELFARE", fontsize=10, fontname="helv")
p2.insert_text((140, 105), "BIRTH CERTIFICATE", fontsize=14, fontname="helv", color=(0.1, 0.3, 0.6))
p2.insert_text((40, 135), "Issued under Registration of Births & Deaths Act", fontsize=9, fontname="helv")
p2.insert_text((40, 170), "Child Name: Aarav Sharma", fontsize=11, fontname="helv")
p2.insert_text((40, 195), "Date of Birth: 15/08/2008 | Sex of Child: Male", fontsize=10, fontname="helv")
p2.insert_text((40, 220), "Father Name: Rajesh Sharma", fontsize=10, fontname="helv")
p2.insert_text((40, 245), "Mother Name: Sunita Sharma", fontsize=10, fontname="helv")
p2.insert_text((40, 275), "Place of Birth: Bengaluru Municipal Corporation", fontsize=10, fontname="helv")
doc_birth.save(SAMPLES_DIR / "sample_birth_certificate.pdf")

# 3. Sample Transfer Certificate (TC) PDF
doc_tc = fitz.open()
p3 = doc_tc.new_page(width=450, height=320)
p3.draw_rect(fitz.Rect(15, 15, 435, 305), color=(0.3, 0.3, 0.3), width=2)
p3.insert_text((120, 50), "ST. JOSEPH HIGH SCHOOL", fontsize=14, fontname="helv")
p3.insert_text((130, 80), "TRANSFER CERTIFICATE (TC)", fontsize=13, fontname="helv", color=(0.6, 0.1, 0.1))
p3.insert_text((40, 120), "TC No: TC/2026/842 | Admission No: 14820", fontsize=10, fontname="helv")
p3.insert_text((40, 150), "Pupil Name: Aarav Sharma", fontsize=11, fontname="helv")
p3.insert_text((40, 180), "Class in which pupil last studied: Grade 10", fontsize=10, fontname="helv")
p3.insert_text((40, 210), "Institution last attended: St. Joseph High School", fontsize=10, fontname="helv")
p3.insert_text((40, 240), "Conduct and Character: Good", fontsize=10, fontname="helv")
p3.insert_text((40, 270), "Date of Leaving: 25/03/2026", fontsize=10, fontname="helv")
doc_tc.save(SAMPLES_DIR / "sample_transfer_certificate.pdf")

# 4. Sample Electricity Bill (Address Proof) PDF
doc_bill = fitz.open()
p4 = doc_bill.new_page(width=450, height=300)
p4.draw_rect(fitz.Rect(15, 15, 435, 285), color=(0.2, 0.5, 0.2), width=1.5)
p4.insert_text((120, 45), "BESCOM ELECTRICITY BILL", fontsize=13, fontname="helv", color=(0.1, 0.5, 0.2))
p4.insert_text((40, 80), "Consumer No: 9876543210 | Bill Date: 10/09/2026", fontsize=10, fontname="helv")
p4.insert_text((40, 110), "Customer Name: Rajesh Sharma", fontsize=11, fontname="helv")
p4.insert_text((40, 140), "Address: #42, 5th Cross, Indiranagar, Bengaluru - 560038", fontsize=10, fontname="helv")
p4.insert_text((40, 175), "Units Consumed: 185 kWh", fontsize=10, fontname="helv")
p4.insert_text((40, 205), "Total Amount Payable: Rs 1,420.00", fontsize=12, fontname="helv", color=(0.7, 0.1, 0.1))
doc_bill.save(SAMPLES_DIR / "sample_utility_bill.pdf")

print("Generated sample PDFs in samples/")
