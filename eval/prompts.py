def xml_prompt(q):
    return f"""
You must solve the problem using the structure below.

<guideline>
Solve the problem carefully.
<plan>
Describe the approach.
</plan>
<step>
Work through the calculation.
</step>
<takeaway>
Give the final numeric answer.
</takeaway>
</guideline>

Problem:
{q}
"""

def natural_prompt(q):
    return f"""
Solve the problem using this structure:

Guideline:
Plan:
Step:
Takeaway:

Problem:
{q}
"""

def glyph_prompt(q):
    return f"""
Solve the problem using the glyph structure:

🜞 Solve the problem carefully.
🜆 Describe the approach.
🜂 Work through the calculation.
🜃 Give the final numeric answer.

Problem:
{q}
"""
