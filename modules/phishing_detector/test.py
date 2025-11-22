# import os

# folders = {
#     "Easy Ham": r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\CYBERCRIME\data\phishing_emails\easy_ham",
#     "Hard Ham": r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\CYBERCRIME\data\phishing_emails\hard_ham",
#     "Spam": r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\CYBERCRIME\data\phishing_emails\spam"
# }

# print("Email Count per Folder:\n" + "-"*30)

# for label, path in folders.items():
#     try:
#         num_files = len([
#             f for f in os.listdir(path)
#             if os.path.isfile(os.path.join(path, f))
#         ])
#         print(f"{label}: {num_files} emails")
#     except Exception as e:
#         print(f"{label}: Error - {e}")

  