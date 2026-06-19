


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


westher_all = "<html>\n\n<head>\n<meta http-equiv=\"Content-Type\"\ncontent=\"text\/html; charset=iso-8859-1\">\n<meta name=\"GENERATOR\" content=\"Microsoft FrontPage Express 2.0\">\n<title>Esther's homepage met Afrikaanse recepten<\/title>\n<\/head>\n\n<body background=\"ringband.gif\" bgcolor=\"#FFFFFF\"\nleftmargin=\"100\">\n\n<p><font face=\"Arial\">Esther's homepage met Afrikaanse recepten<\/font><\/p>\n\n<p><font face=\"Arial\">Op deze homepage staan Afrikaanse recepten\nin het Nederlands, Engels, Duits en Frans. Ik heb geprobeerd bij\nelk recept zo goed mogelijk aan te geven wie de bedenker ervan\nis. Mocht je een van je eigen recepten op deze pagina's vinden,\nen ben je het er niet mee eens dat het op mijn pagina staat, laat\nme dat dan weten, dan haal ik het recept er vanaf.<\/font><\/p>\n\n<p><font face=\"Arial\">De recepten zijn op drie manieren te vinden\n:<\/font><\/p>\n\n<p><font face=\"Arial\">- door in de <\/font><a\nhref=\"http:\/\/www.xs4all.nl\/~westher\/recepten\/\"><font face=\"Arial\">lijst<\/font><\/a><font\nface=\"Arial\"> van alle recepten te kijken<\/font><\/p>\n\n<p><font face=\"Arial\">- door gebruik te maken van het zoekvenster<\/font><\/p>\n\n<p><font face=\"Arial\">Typ een woord in het zoekvenster, klik op\n&lt;Search&gt; en alle recepten waarin dat woord gebruikt wordt,\nworden getoond.<\/font><\/p>\n\n<p><font face=\"Arial\"><!--webbot bot=\"HTMLMarkup\" startspan --><!-- Begin PicoSearch Code -->\n<P><FORM METHOD=\"POST\" ACTION=\"http:\/\/www.picosearch.com\/cgi-bin\/ts.pl\">\n<INPUT TYPE=\"HIDDEN\" NAME=\"index\" VALUE=\"72503\">\n<TABLE BGCOLOR=\"WHITE\" CELLSPACING=0 CELLPADDING=0 BORDER=0><TR><TD>\n<TABLE BGCOLOR=\"WHITE\" CELLSPACING=2 CELLPADDING=0 BORDER=0>\n<TR><TD><A HREF=\"http:\/\/www.picosearch.com\/\">\n<IMG BORDER=\"0\" SRC=\"http:\/\/www.picosearch.com\/picosmall.gif\" ALT=\"PicoSearch\"><\/A><\/TD>\n<TD><INPUT TYPE=\"TEXT\" NAME=\"query\" VALUE=\"\" SIZE=\"20\"><\/TD>\n<TD><INPUT TYPE=\"SUBMIT\" VALUE=\"Search\"><\/TD><\/TR>\n<\/TABLE><\/TD><\/TR><\/TABLE><\/FORM>\n<!-- End PicoSearch Code --> \n<!--webbot\nbot=\"HTMLMarkup\" endspan --><\/font><\/p>\n\n<p><font face=\"Arial\">- door gebruik te maken van de <\/font><a\nhref=\"http:\/\/www.xs4all.nl\/~westher\/thema's\/landenpagina.htm\"><font\nface=\"Arial\">landenpagina<\/font><\/a><\/p>\n\n<p><font face=\"Arial\">Heb je opmerkingen over deze site, mail dan\nnaar <\/font><a href=\"mailto:westher@xs4all.nl\"><font face=\"Arial\">Esther\nWesterveld<\/font><\/a><font face=\"Arial\">.<\/font><\/p>\n\n<p><font face=\"Arial\"><!--webbot bot=\"HTMLMarkup\" startspan --><center>\n<a href=\"http:\/\/groups.yahoo.com\/group\/AFRIKAANSErecepten\/join\">\n<img src=\"http:\/\/us.i1.yimg.com\/us.yimg.com\/i\/yg\/img\/i\/us\/ui\/join.gif\" border=\"0\"\n  alt=\"Klik om lid te worden van AFRIKAANSErecepten\"><br>Klik om lid te worden van AFRIKAANSErecepten<\/a>\n<\/center><!--webbot\nbot=\"HTMLMarkup\" endspan --><\/font><!--webbot bot=\"HTMLMarkup\"\nstartspan --><script type=\"text\/javascript\">\nvar gaJsHost = ((\"https:\" == document.location.protocol) ? \"https:\/\/ssl.\" : \"http:\/\/www.\");\ndocument.write(unescape(\"%3Cscript src='\" + gaJsHost + \"google-analytics.com\/ga.js' type='text\/javascript'%3E%3C\/script%3E\"));\n<\/script>\n<script type=\"text\/javascript\">\nvar pageTracker = _gat._getTracker(\"UA-3585454-2\");\npageTracker._initData();\npageTracker._trackPageview();\n<\/script><!--webbot bot=\"HTMLMarkup\" endspan --><\/p>\n<\/body>\n<\/html>\n"

westher_html = "<html>\n\n<head>\n<meta http-equiv=\"Content-Type\"\ncontent=\"text\/html; charset=iso-8859-1\">\n<meta name=\"GENERATOR\" content=\"Microsoft FrontPage Express 2.0\">\n<title>Esther's homepage met Afrikaanse recepten<\/title>\n<\/head>\n\n<body background=\"ringband.gif\" bgcolor=\"#FFFFFF\"\nleftmargin=\"100\">\n\n<p><font face=\"Arial\">Esther's homepage met Afrikaanse recepten<\/font><\/p>\n\n<p><font face=\"Arial\">Op deze homepage staan Afrikaanse recepten\nin het Nederlands, Engels, Duits en Frans. Ik heb geprobeerd bij\nelk recept zo goed mogelijk aan te geven wie de bedenker ervan\nis. Mocht je een van je eigen recepten op deze pagina's vinden,\nen ben je het er niet mee eens dat het op mijn pagina staat, laat\nme dat dan weten, dan haal ik het recept er vanaf.<\/font><\/p>\n\n<p><font face=\"Arial\">De recepten zijn op drie manieren te vinden\n:<\/font><\/p>\n\n<p><font face=\"Arial\">- door in de <\/font><a\nhref=\"http:\/\/www.xs4all.nl\/~westher\/recepten\/\"><font face=\"Arial\">lijst<\/font><\/a><font\nface=\"Arial\"> van alle recepten te kijken<\/font><\/p>\n\n<p><font face=\"Arial\">- door gebruik te maken van het zoekvenster<\/font><\/p>\n\n<p><font face=\"Arial\">Typ een woord in het zoekvenster, klik op\n&lt;Search&gt; en alle recepten waarin dat woord gebruikt wordt,\nworden getoond.<\/font><\/p>\n\n<p><font face=\"Arial\"><!--webbot bot=\"HTMLMarkup\" startspan --><!-- Begin PicoSearch Code -->\n<P><FORM METHOD=\"POST\" ACTION=\"http:\/\/www.picosearch.com\/cgi-bin\/ts.pl\">\n<INPUT TYPE=\"HIDDEN\" NAME=\"index\" VALUE=\"72503\">\n<TABLE BGCOLOR=\"WHITE\" CELLSPACING=0 CELLPADDING=0 BORDER=0><TR><TD>\n<TABLE BGCOLOR=\"WHITE\" CELLSPACING=2 CELLPADDING=0 BORDER=0>\n<TR><TD><A HREF=\"http:\/\/www.picosearch.com\/\">\n<IMG BORDER=\"0\" SRC=\"http:\/\/www.picosearch.com\/picosmall.gif\" ALT=\"PicoSearch\"><\/A><\/TD>\n<TD><INPUT TYPE=\"TEXT\" NAME=\"query\" VALUE=\"\" SIZE=\"20\"><\/TD>\n<TD><INPUT TYPE=\"SUBMIT\" VALUE=\"Search\"><\/TD><\/TR>\n<\/TABLE><\/TD><\/TR><\/TABLE><\/FORM>\n<!-- End PicoSearch Code --> \n<!--webbot\nbot=\"HTMLMarkup\" endspan --><\/font><\/p>\n\n<p><font face=\"Arial\">- door gebruik te maken van de <\/font><a\nhref=\"http:\/\/www.xs4all.nl\/~westher\/thema's\/landenpagina.htm\"><font\nface=\"Arial\">landenpagina<\/font><\/a><\/p>\n\n<p><font face=\"Arial\">Heb je opmerkingen over deze site, mail dan\nnaar <\/font><a href=\"mailto:westher@xs4all.nl\"><font face=\"Arial\">Esther\nWesterveld<\/font><\/a><font face=\"Arial\">.<\/font><\/p>\n\n<p><font face=\"Arial\"><!--webbot bot=\"HTMLMarkup\" startspan --><center>\n<a href=\"http:\/\/groups.yahoo.com\/group\/AFRIKAANSErecepten\/join\">\n<img src=\"http:\/\/us.i1.yimg.com\/us.yimg.com\/i\/yg\/img\/i\/us\/ui\/join.gif\" border=\"0\"\n  alt=\"Klik om lid te worden van AFRIKAANSErecepten\"><br>Klik om lid te worden van AFRIKAANSErecepten<\/a>\n<\/center><!--webbot\nbot=\"HTMLMarkup\" endspan --><\/font><!--webbot bot=\"HTMLMarkup\"\nstartspan --><script type=\"text\/javascript\">\nvar gaJsHost = ((\"https:\" == document.location.protocol) ? \"https:\/\/ssl.\" : \"http:\/\/www.\");\ndocument.write(unescape(\"%3Cscript src='\" + gaJsHost + \"google-analytics.com\/ga.js' type='text\/javascript'%3E%3C\/script%3E\"));\n<\/script>\n<script type=\"text\/javascript\">\nvar pageTracker = _gat._getTracker(\"UA-3585454-2\");\npageTracker._initData();\npageTracker._trackPageview();\n<\/script><!--webbot bot=\"HTMLMarkup\" endspan --><\/p>\n<\/body>\n<\/html>\n"

print(len(westher_all), len(westher_html))