"""Builds Datasets/challenge_set.csv (hand-written hard cases) and Datasets/university_test.csv
(real-world test from Datasets final.csv, column -> type mapping, 500 samples, seed 42).
Run: python PyScripts\\build_eval_sets.py
"""
import pandas as pd
from blueedge_common import CHALLENGE, UNI_RAW, UNI_TEST

HARD = {
 "phone_number": ["1234567890", "03001234567", "+92 300 1234567", "(212) 555-0147", "+44 20 7946 0958",
                  "0092-51-2345678", "555.867.5309", "+1-800-555-0199", "051 9261234", "+91 98765 43210",
                  "300-1234567", "0300 1234567", "+49 30 901820", "(021) 3456789"],
 "numeric_id": ["ID-123-456", "student_12345", "123456789", "STU2023001", "A-998877", "20231045", "0012345",
                "EMP-4455", "990011223344", "REG#7788", "ID 556677", "2023-CS-114", "624353", "10293847"],
 "ssn": ["123-45-6789", "987-65-4321", "078-05-1120", "219 09 9999", "001-01-0001", "666-12-3456",
         "123 45 6789", "555-44-3333", "XXX-XX-6789", "***-**-4321", "111-22-3333", "900-70-0000",
         "321-54-9876", "123456789"],
 "date": ["2023/12/25", "12/25/2023", "25-12-2023", "2023-12-25", "25 Dec 2023", "Dec 25, 2023", "25-Dec-23",
          "2023年12月25日", "1/2/99", "31.01.2024", "2024.03.15", "15-Mar-2024", "03-15-2024", "Jan 5 2022"],
 "email": ["john.doe@mail.com", "user+tag@gmail.co.uk", "a_b-c@sub.domain.org", "student12@uni.edu.pk",
           "USER@EXAMPLE.COM", "x@y.io", "first.last@company-name.com", "info@my.site.net",
           "name123@hotmail.com", "test_user@yahoo.in", "ali.khan@nust.edu.pk", "mail@domain.travel",
           "u.s.e.r@e.co", "support@shop.example.com"],
 "url": ["https://example.com", "www.example.org", "http://sub.domain.com/path?q=1", "ftp://files.server.net/a.zip",
         "https://github.com/user/repo", "example.com/page", "www.nust.edu.pk", "https://a.b.c.d/e/f",
         "http://192.168.0.1:8080/login", "https://www.google.com/search?q=test", "www.site.io/x",
         "https://t.co/abc123", "http://localhost:3000", "https://en.wikipedia.org/wiki/Edge_computing"],
 "currency": ["$1,234.56", "€500", "£99.99", "¥10000", "₨ 25,000", "Rs. 5000", "USD 250", "$ 12", "1,000 PKR",
              "€ 1.234,50", "$0.99", "₨1,50,000", "£1,000,000", "¥ 500"],
 "percentage": ["45%", "12.5 %", "100%", "0.5%", "fifty percent", "7 percent", "99.9%", "3.14%", "20 percent",
                "-5%", "15%", "8.25 %", "60%", "33.3%"],
 "binary": ["true", "False", "YES", "no", "Y", "n", "1", "0", "active", "Inactive", "ON", "off", "T", "F"],
 "categorical": ["red", "Male", "Female", "Single", "Premium", "Level 3", "Category B", "Gold", "Pending",
                 "Approved", "Medium", "Graduate", "Sedan", "Other"],
 "geographic_coordinates": ["33.6844, 73.0479", "-33.8688, 151.2093", "40.7128,-74.0060", "51.5074, -0.1278",
                            "35.6762, 139.6503", "-1.2921, 36.8219", "48.8566, 2.3522", "24.8607, 67.0011",
                            "0.0, 0.0", "55.7558, 37.6173", "-22.9068, -43.1729", "31.5204, 74.3587",
                            "19.4326, -99.1332", "1.3521, 103.8198"],
 "address": ["123 Main Street", "45 Park Ave, Apt 4B", "House 12, Street 5, F-8/3, Islamabad", "PO Box 1234",
             "221B Baker Street, London", "Flat 7, Block C, Gulberg III", "1600 Pennsylvania Avenue NW",
             "78 Elm Road", "Unit 5, 99 King St", "Plot 22, Sector G-10", "10 Downing Street",
             "350 Fifth Avenue, New York", "Street 9, Satellite Town, Rawalpindi", "42 Wallaby Way, Sydney"],
 "person_name": ["John Smith", "Muhammad Ali Khan", "Dr. Sara Ahmed", "Mary-Jane O'Neil", "Ayesha Siddiqui",
                 "Prof. Haitham El-ghareeb", "Wei Zhang", "Jean-Luc Picard", "Fatima Bibi", "Robert James Smith Jr.",
                 "Mr. Bilal Hussain", "Nagwa Elmobark", "Li Na", "Omar Farooq"],
 "text_description": ["This product is great for daily use", "Meeting scheduled for next week to discuss the budget",
                      "Customer reported a delay in shipment", "Lorem ipsum dolor sit amet",
                      "The quick brown fox jumps over the lazy dog", "Please contact support if the issue persists",
                      "Battery lasts about ten hours", "Not working after update",
                      "Room 5 is on the second floor near the elevator", "Great service, will come again",
                      "Order #1234 was cancelled by the user", "Handle with care", "A short note",
                      "Delivered at 5 pm yesterday, signed by reception"],
}

rows = [(s, t) for t, lst in HARD.items() for s in lst]
pd.DataFrame(rows, columns=["sample", "data_type"]).to_csv(CHALLENGE, index=False, encoding="utf-8")
print(f"Challenge set: {len(rows)} samples -> {CHALLENGE}  (paper: 200; extend HARD dict to reach 200)")

# ------------------------------------------------------------------ university real-world test
MAP = {"Full Name": "person_name", "Email": "email", "Mobile": "phone_number", "Birthday": "date",
       "ID Number": "numeric_id", "Gender": "categorical", "Status": "categorical"}
raw = pd.read_csv(UNI_RAW, encoding="cp1252", dtype=str)
raw.columns = [c.strip() for c in raw.columns]
parts = []
for col, typ in MAP.items():
    if col not in raw.columns:
        print("missing column:", col); continue
    s = raw[col].dropna().astype(str).str.strip()
    s = s[s != ""]
    parts.append(pd.DataFrame({"sample": s, "data_type": typ, "source_col": col}))
pool = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["sample", "data_type"])
# balance by type: ~500 total across 6 types (categorical has 2 source cols, capped together)
per = 500 // pool["data_type"].nunique()
test = (pool.groupby("data_type", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per), random_state=42)).reset_index(drop=True))
test.to_csv(UNI_TEST, index=False, encoding="utf-8")
print(f"University test: {len(test)} samples -> {UNI_TEST}")
print(test["data_type"].value_counts().to_string())
print("NOTE: labels derive from column names (weak labels). Spot-check ~50 rows manually (paper: 500 manually labeled).")
print("NOTE: Mobile read as string. If leading zeros already lost in CSV, phone samples look like numeric IDs.")
