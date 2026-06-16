


# # 20190827112156

# import requests

# API_URL = "https://api-inference.huggingface.co/models/anoukflinkert/time2warc-roberta"
# headers = {"Authorization": "Bearer hf_YOUR_ACTUAL_TOKEN_HERE"} 

# print("Pinging Hugging Face API...")
# try:
#     # The timeout prevents it from hanging forever
#     response = requests.post(
#         API_URL, 
#         headers=headers, 
#         json={"inputs": "Welcome to my retro 1990s web page!"}, 
#         timeout=30 
#     )
#     print(f"Status Code: {response.status_code}")
#     print(f"Response: {response.text}")
# except Exception as e:
#     print(f"Connection Failed: {e}")

import streamlit as st
import pandas as pd
import re

# df = pd.read_json('./output/time2warc_final_predictions.json', lines=False)
# Example generation logic
# base_url = "http://webarchief.kb.nl:8080/archived/"

# http://webarchief.kb.nl:8080/archived/20190827112654/https://pcvdklis.home.xs4all.nl/

# IAH-20191016104014764-00000-18666~webharvest-app02.mw.prod.bibliotheek.lcl~8443.warc
# http://webarchief.kb.nl:8080/archived/20191016104015/https://rongen17.home.xs4all.nl/

# base_url = "http://webarchief.kb.nl:8080/archived/"
# timestamp = df['warc_filename'].str.extract(r'IAH-(\d{14})', expand=False)

# df['wayback_link'] = base_url + timestamp + "/https://" + df['seed_url'].astype(str)
# print(df.iloc[0]['wayback_link'])

# # Displaying it as a clickable link in the Streamlit dataframe
# st.dataframe(
#     df,
#     column_config={
#         "wayback_link": st.column_config.LinkColumn("Open in Wayback Machine")
#     }
# )




# # # --- CUSTOM SQL EXPLORATION ---
#             st.markdown("---")
#             st.subheader("Custom SQL Workspace")
#             st.write("Write raw SQLite queries to filter, aggregate, and explore your parsed data.")
            
#             user_query = st.text_area("SQL Query", value="SELECT period, COUNT(*) as count FROM websites GROUP BY period ORDER BY count DESC;")
            
#             if st.button("Execute SQL"):
#                 try:
#                     custom_df = pd.read_sql_query(user_query, conn)
#                     st.dataframe(custom_df, use_container_width=True, hide_index=True)
                    
#                     # Provide an instant CSV download for whatever they queried
#                     st.download_button(
#                         label="Download Query Results (CSV)",
#                         data=custom_df.to_csv(index=False).encode('utf-8'),
#                         file_name="custom_sql_export.csv",
#                         mime="text/csv"
#                     )
#                 except Exception as e:
#                     st.error(f"SQL Error: {e}")


t = "<html>\n\n<head>\n<meta http-equiv=\"Content-Type\"\ncontent=\"text/html; charset=iso-8859-1\">\n<meta http-equiv=\"Content-Type\"\ncontent=\"text/html; charset=iso-8859-1\">\n<meta http-equiv=\"reply-to\" content=\"pcvdklis@xs4all.nl\">\n<meta http-equiv=\"refresh\"\ncontent=\"0; URL=http://www.xs4all.nl/~pcvdklis/atelie/stainedglassatelier.htm\">\n<meta name=\"description\" content=\"weblog van der Klis\">\n<meta name=\"keywords\" content>\n<meta name=\"author\" content=\"pcvdklis@xs4all.nl\">\n<meta name=\"publisher\" content=\"pcvdklis\">\n<meta name=\"language\" content=\"EN, NL\">\n<meta name=\"robots\" content=\"ALL\">\n<meta name=\"GENERATOR\" content=\"Microsoft FrontPage Express 2.0\">\n<title></title>\n<!-- <bgsound src=\"\" loop=\"\"> -->\n</head>\n\n<body bgcolor=\"#EAF7C4\">\n<div align=\"center\"><center>\n\n<pre><!--webbot bot=\"HTMLMarkup\" startspan --> <!--webbot\nbot=\"HTMLMarkup\" endspan --><font size=\"6\" face=\"Times New Roman\"> </font></pre>\n</center></div>\n</body>\n</html>\n\n\n<html>\n\n<head>\n<meta http-equiv=\"Content-Type\"\ncontent=\"text/html;"


t2 = '<html>\n\n<head>\n<meta http-equiv=\"Content-Type\"\ncontent=\"text\/html; charset=iso-8859-1\">\n<meta http-equiv=\"Content-Type\"\ncontent=\"text\/html; charset=iso-8859-1\">\n<meta http-equiv=\"reply-to\" content=\"pcvdklis@xs4all.nl\">\n<meta http-equiv=\"refresh\"\ncontent=\"0; URL=http:\/\/www.xs4all.nl\/~pcvdklis\/atelie\/stainedglassatelier.htm\">\n<meta name=\"description\" content=\"weblog van der Klis\">\n<meta name=\"keywords\" content>\n<meta name=\"author\" content=\"pcvdklis@xs4all.nl\">\n<meta name=\"publisher\" content=\"pcvdklis\">\n<meta name=\"language\" content=\"EN, NL\">\n<meta name=\"robots\" content=\"ALL\">\n<meta name=\"GENERATOR\" content=\"Microsoft FrontPage Express 2.0\">\n<title><\/title>\n<!-- <bgsound src=\"\" loop=\"\"> -->\n<\/head>\n\n<body bgcolor=\"#EAF7C4\">\n<div align=\"center\"><center>\n\n<pre><!--webbot bot=\"HTMLMarkup\" startspan --> <!--webbot\nbot=\"HTMLMarkup\" endspan --><font size=\"6\" face=\"Times New Roman\"> <\/font><\/pre>\n<\/center><\/div>\n<\/body>\n<\/html>\n'

print(len(t), len(t2))