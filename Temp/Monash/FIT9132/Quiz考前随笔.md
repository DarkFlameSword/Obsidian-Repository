---
date: 2025-05-29
author:
  - Siyuan Liu
tags:
  - FIT9132
aliases:
  - 随笔
---
# Key Word
IS NULL
NOT NULL
# Function
COUNT(\*): include NULL
COUNT(1): exclude NULL
COUNT(column name): exclude NULL
NVL(value1, value2): attention that keep consistency between value1 and value 2
TO_CHAR(): TO_CHAR(2344.54342, $'9999.99')
TO_DATE(): TO_DATE('2025/5/29 15:43:23', 'YYYY/MM/DD HH24:MI:SS')

# ATTENTION
the alias in SELECT clause should only be used in ORDER BY and some LIMIT clauses
the alias in FROM clause could be used in everywhere
