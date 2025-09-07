# Credit: this code is adapted from https://github.com/lapisrocks/LanguageAgentTreeSearch

prompt_cot_no_think_selection = """You are a web agent, select the best next action for the search to fulfill the task. Example tasks are shown below.
Provide a rationale for your selection BEFORE you provide the action.
NOTE: You can only select actions that are provided in the Possible Actions list. You MAY NOT select actions in the Not Allowed list.
NOTE: You must output BOTH a rationale and an action.
Note: Do not select any of the following actions: 'Back to Search', 'Next >', '< Prev', 'Attributes', 'Description', 'Features', 'Reviews', even if they are available on the page.
Example #1:
Webshop 
Instruction:  
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[Search]

Action: search[3 ounce bright citrus deodorant sensitive skin < 50 dollars]
Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

Actions Not Allowed:
click[B08KBVJ4XN]
click[< Prev]
click[Back to Search]

Possible Next Actions:
click[B078GTKVXY]
click[B078GWRC1J]
click[Next >]

Rationale: The next action is to click 'Bright Citrus Deodorant by Earth Mama' (B078GWRC1J) as it seems to meet all the specified criteria: 3 ounce bottle, bright citrus, suitable for sensitive skin, and priced under $50.00.
Action: click[B078GWRC1J]

Example #2:
Webshop 
Instruction:  
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[Search]

Action: search[3 ounce bright citrus deodorant for sensitive skin < 50]
Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

Action: click[B078GWRC1J]
Observation: 
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Buy Now] 

Actions Not Allowed:
click[3 ounce (pack of 1)]
click[3-ounce (2-pack)]
click[< Prev]
click[Back to Search]
click[Buy Now]
click[Description]
click[Features]
click[Reviews]
click[calming lavender]
click[ginger fresh]
click[simply non-scents]
click[travel set (4-pack)]

Possible Next Actions:
click[assorted scents]
click[bright citrus]

Rationale: The best next action is to select 'bright citrus' as the scent in order to match the specified criteria.
Action: click[bright citrus]
-----------------------------------
New Task:
{task}

Actions Not Allowed:
{not_allowed_actions}

Possible Next Actions (REMINDER: You can ONLY!!! select actions from this list.):
{possible_actions}
REMINDER: Do not select any of the following actions: 'Back to Search', 'Next >', '< Prev', 'Attributes', 'Description', 'Features', 'Reviews', even if they are available on the page.

"""
prompt_cot_no_think = """You are a web agent, provide the next action for the search to fulfill the task. An example task is shown below. Your action must start with 'search' or 'click'.
Note if you want to search with a new query you need to click Back to Search.
For products on the search page, to click on them use their code i.e. click[B078GWRC1J], click[B078GTKVXY], etc.
NOTE: For options on the product page, to click on them use their names i.e. click[bright citrus], click[3 ounce (pack of 1)], click[large], click[red], etc. 
IMPORTANT: The option you indicate must exactly match the option text on the page.
NOTE: it is vital that you consider selecting these options as they are often crucial to be able to match the task requirements.
NOTE: Do not repeat any actions that have already been taken in the trajectory.
NOTE: The trajectory only includes the instruction, the only possible action is to search.
NOTE: Search queries should include cost restriction / other restrictions.
You will provide a rationale BEFORE you provide the action.

Example #1:
Webshop 
Instruction:  
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[Search]

What might be the next action? 
Rationale: The next action is to search for a 3 ounce bright citrus deodorant suitable for sensitive skin and priced under 50.00 dollars.
Action: search[bright citrus deodorant sensitive skin 3 ounce  bottle]

Example #2:
Webshop 
Instruction:  
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[Search]

Action: search[3 ounce bright citrus deodorant sensitive skin < 50 dollars]
Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

What might be the next action? 
Rationale: The next action is to select 'Bright Citrus Deodorant by Earth Mama' as it meets all the specified criteria: 3 ounce bottle, bright citrus, suitable for sensitive skin, and priced under $50.00.
Action: click[B078GWRC1J]

Example #3:
Webshop 
Instruction:  
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[Search]

Action: search[3 ounce bright citrus deodorant for sensitive skin < 50]
Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

Action: click[B078GWRC1J]
Observation: 
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Buy Now] 

What might be the next action?
Rationale: The next action is to select 'bright citrus' as the scent in order to match the specified criteria.
Action: click[bright citrus]
-----------------------------------
New Task:
{input}
What might be the next action?
"""


prompt_cot_feedback = """You are also an advanced reasoning agent that can improve based on self refection. Follow the instruction and purchase an item meeting all of the correct specifications by navigating the website. 
Here is an example:
Webshop 
Instruction:  
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[Search]  

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

Action: think[B078GWRC1J and B078GTKVXY are bright citrus deodorant less then 50 dollars. I can check B078GWRC1J first.]
Observation: OK.

Action: click[B078GWRC1J]
Observation: 
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Buy Now]  

Action: think[For 3 ounce bottle of bright citrus deodorant for sensitive skin, the item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy.]
Observation: OK.

Action: click[bright citrus]
Observation: You have clicked bright citrus. 

Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1). 

Action: click[Buy Now]

Here are some previous trials in which you were given access to relevant context and a item to purchase. You were unsuccessful in buying the correct item either because you did not find an item meeting all of the required specifications or because you did not select the correct item. Carefully consider the errors and make sure not to repeat the same mistakes. {trajectories}

{input}
"""

score_prompt_finetuned= '''Determine the value / correctness score (1 - 10) of the current state.
You must provide the best next action and the observation (in the same format as the current observation) for this best successor state. Then, you must provide a reflection on the best successor state and derive the correctness score.

Trajectory: 
{truncated_trajectory}
Current Action: {current_action}
Current Observation: {current_observation}
'''

score_prompt_finetuned_terminal = '''Determine the value / correctness score (1 - 10) of the terminal state.
You must provide a reflection on the terminal state and determine the correctness score (do not use the discount factor).
Trajectory:
{truncated_trajectory}
Current Action: {current_action}
Current Observation: {current_observation}
'''

'''
1: The last action and observed state captures the purchase of an item that DOES NOT MATCH ANY of the attributes specified in the task.
5: The last action and observed state captures the purchase of an item that has SOME of the attributes matching the specifications.
7: The last action and observed state captures the purchase of an item that has MOST of the attributes matching the specifications.
10: The last action and observed state captures the successful purchase of the item that FULLY AND EXACTLY matches the specifications.

'''

score_prompt_terminal = '''Given an item to purchase and a trajectory that aims to buy an item that exactly matches the specification, analyze the the correctness of the item purchased.
Provide a reflection that concludes with. "Thus the correctness score is s", where s is either 1, 5, 7, or 10.  Use the following scale for scoring:

1: The last action and observed state captures the purchase of an item that DOES NOT MATCH ANY of the attributes specified in the task.
5: The last action and observed state captures the purchase of an item that has SOME of the attributes matching the specifications.
7: The last action and observed state captures the purchase of an item that has MOST of the attributes matching the specifications.
10: The last action and observed state captures the successful purchase of the item that FULLY AND EXACTLY matches the specifications.

Keep reflections short (< 100 words). The reflection should enumerate the attributes in the task description and evaluate whether or not the selected item meets each criteria. These criteria might include but are not limited to the item's type, name, size, flavor, color, price, etc.
Follow the format of the rationale from the below example task. Output [SUCCESS] if the item meets the criteria and [FAILURE] if it does not for each attribute. No other options are allowed.

Reminder: ONLY SELECT 10 if the item EXACTLY matches the specifications.
Example Task #1:

Webshop
Instruction:
i am looking for dairy free and apple variety pack of chips, and price lower than 30.00 dollars
[Search]

Action: search[dairy free and apple variety pack of chips]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B07HRFJWP8]
Enjoy Life Foods Soft Baked Ovals, Breakfast Bars, Nut Free Bars, Soy Free, Dairy Free, Non GMO, Gluten Free, Vegan, Variety Pack, 4 Boxes (20 Bars Total)
$100.0
[B01KMHY5PG]
Enjoy Life Soft Baked Chewy Bars, Variety Pack, Nut Free Bars, Soy Free, Dairy Free, Gluten Free, 6 Boxes (30 Total Bars)
$21.49
[B008D2X8C4]
Enjoy Life Lentil Chips Variety Pack, Dairy Free Chips, Soy Free, Nut Free, Non GMO, Vegan, Gluten Free, 24 Bags (0.8 oz)
$100.0

Action: click[B008D2X8C4]
Observation:
[Back to Search]
[< Prev]
flavor name [dill and sour cream][garlic & parmesan][light sea salt][margherita pizza][thai chili lime][variety pack]
size [0.8 ounce (pack of 24)][4 ounce (pack of 12)]
Price: $100.0
Rating: N.A.
[Description]
[Features]
[Reviews]
[Buy Now]

Action: click[variety pack]
Observation: You have clicked variety pack.

Action: click[0.8 ounce (pack of 24)]
Observation: You have clicked 0.8 ounce (pack of 24).

Action: click[Buy Now]

Reflection: The purchased item is analyzed against the task specification as follows:
1. Dairy-Free: The item matches this attribute. [SUCCESS]
2. Apple Variety Pack of Chips: The item is a variety pack of chips but does not specify "apple" as a variety, failing this condition. [FAILURE]
3. Price Check: The price limit is $30. The item is priced at $100. Since $100 > $30, the product exceeds the price limit. [FAILURE]
Thus, one of the three checks were successful. Therefore the item meets less than half of the specified criteria.
Thus the correctness score is 5.
Example Task #2:
Webshop 
Instruction:  
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[Search]  

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

Action: click[B078GWRC1J]
Observation: 
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Buy Now]

Action: click[bright citrus]
Observation: You have clicked bright citrus.

Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1).

Action: click[Buy Now]

Reflection: The purchased item is analyzed against the task specification as follows:
1. 3 Ounce Bottle: The item matches this attribute. [SUCCESS]
2. Bright Citrus Deodorant: The item matches this attribute. [SUCCESS]
3. Suitable for Sensitive Skin: The item matches this attribute. [SUCCESS]
4. Price Check: The price limit is $50. The item is priced at $10.99. Since $10.99 < $50, the product is within the price limit. [SUCCESS]
Thus, all of the specified criteria are met. The item exactly matches the specifications.
Thus the correctness score is 10.

New Task (Reminder: your response MUST have either a [SUCCESS] or [FAILURE] tag for each attribute and should end with "Thus the correctness score is ...")
{input}

Reflection:
'''


score_prompt = '''Given an item to purchase and a trajectory that aims to buy an item that exactly matches the specification, analyze the how well the last action and observation aligns with the task.
Provide a reflection that concludes with. "Thus the correctness score is s", where s is either 1, 2, 4, 6, 8, or 10. Use the following scale for scoring:

1: The last action and observed state is entirely irrelevant to the task or captures a purchase of an item that is completely unrelated to the specifications.
2: The last action and observed state captures a step with a low likelihood of leading to purchasing the correct item.
4: The last action and observed state captures a step with a moderate likelihood of leading to purchasing the correct item.
6: The last action and observed state captures a step with a high likelihood of leading to purchasing the correct item.
8: The last action and observed state captures a step with a very high likelihood of leading to purchasing the correct item.
10: The last action and observed state captures a step that will certainly lead to purchasing the correct item.

Keep reflections short (< 100 words). Follow the format of the rationale from the below example task.
NOTE: the observation from clicking on the item will be the item's product detail page. For instance, click[B078GWRC1J] will show the product detail page for the item with code B078GWRC1J which will include the item's name (e.g. Bright Citrus Deodorant by Earth Mama), price ($10.99), and other relevant details as well as options.
NOTE: ASSUME NONE OF THE ATTRIBUTES ON THE PRODUCT PAGE ARE SELECTED
ONLY PROVIDE THE REFLECTION FOR THE LAST ACTION.

Example Task:

Webshop 
Instruction:  
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[Search]  

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

Action: click[B078GWRC1J]
Observation: 
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Buy Now]

Reflection: The last action selects an item B078GWRC1J. 
Based on the trajectory and observation, this product is a deodorant, is suitable for sensitive skin, and has a price of $10.99.
Therefore, this proudct meets some of specified criteria except that is does not yet have a confirmed scent of bright citrus and a confirmed size of 3 ounces.
However, the bright citrus scent and 3 ounce size are available options to select on the product page.
The last action and observation thus capture a step with a high likelihood of leading to purchasing the correct item as only the size attribute is not yet confirmed.
Thus the correctness score is 6.

New Task:
{input}

Respond with the reflection for the last observation of the new task ONLY. As a reminder the last action and observation is as follows:
{last_action}
Your response should start with "Reflection:" and end with "Thus the correctness score is ...".'''


score_prompt_lookahead = '''Given an item to purchase and a trajectory that aims to buy an item that exactly matches the specification, analyze the how well the last action and observation aligns with the task.
Provide a reflection that concludes with. "Thus the correctness score is s", where s is either 1, 2, 4, 6, 8, or 10. Use the following scale for scoring:

1: The last action and observed state is entirely irrelevant to the task or captures a purchase of an item that is completely unrelated to the specifications.
2: The last action and observed state captures a step with a low likelihood of leading to purchasing the correct item.
4: The last action and observed state captures a step with a moderate likelihood of leading to purchasing the correct item.
6: The last action and observed state captures a step with a high likelihood of leading to purchasing the correct item.
8: The last action and observed state captures a step with a very high likelihood of leading to purchasing the correct item.
10: The last action and observed state captures a step that will certainly lead to purchasing the correct item.

Keep reflections short (< 100 words). Follow the format of the rationale from the below example task.
NOTE: the observation from clicking on the item will be the item's product detail page. For instance, click[B078GWRC1J] will show the product detail page for the item with code B078GWRC1J which will include the item's name (e.g. Bright Citrus Deodorant by Earth Mama), price ($10.99), and other relevant details as well as options.
NOTE: ASSUME NONE OF THE ATTRIBUTES ON THE PRODUCT PAGE ARE SELECTED
NOTE: Your reflection should perform a step of LOOKAHEAD where you consider the next best possible action to determine the goodness of the current state.
ONLY PROVIDE THE REFLECTION FOR THE LAST ACTION.

Example Task #1:
Webshop 
Instruction:  
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[Search]  

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

Action: click[B078GWRC1J]
Observation: 
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Buy Now]

Reflection: I will evaluate the best successor state from the current state:
Best Next Action: click[bright citrus]
Observation of Best Successor State: You have clicked bright citrus.
Reflection of the Best Successor State: The last action and observation of clicking on bright citrus does capture a few of the required attributes: the scent, the cost, and the sensitivity. Only the size is not yet confirmed. However, the size is an available option to select on the product page. Therefore, the best successor state captures a step with a very high likelihood of leading to purchasing the correct item.
Thus the correctness score is 8.

Example Task #2:

Webshop
Instruction:
i am looking for dairy free and apple variety pack of chips, and price lower than 30.00 dollars
[Search]

Action: search[dairy free and apple variety pack of chips]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B07HRFJWP8]
Enjoy Life Foods Soft Baked Ovals, Breakfast Bars, Nut Free Bars, Soy Free, Dairy Free, Non GMO, Gluten Free, Vegan, Variety Pack, 4 Boxes (20 Bars Total)
$100.0
[B01KMHY5PG]
Enjoy Life Soft Baked Chewy Bars, Variety Pack, Nut Free Bars, Soy Free, Dairy Free, Gluten Free, 6 Boxes (30 Total Bars)
$21.49
[B008D2X8C4]
Enjoy Life Lentil Chips Variety Pack, Dairy Free Chips, Soy Free, Nut Free, Non GMO, Vegan, Gluten Free, 24 Bags (0.8 oz)
$100.0

Action: click[B008D2X8C4]
Observation:
[Back to Search]
[< Prev]
flavor name [dill and sour cream][garlic & parmesan][light sea salt][margherita pizza][thai chili lime][variety pack]
size [0.8 ounce (pack of 24)][4 ounce (pack of 12)]
Price: $100.0
Rating: N.A.
[Description]
[Features]
[Reviews]
[Buy Now]

Reflection: I will evaluate the best successor state from the current state:
Best Next Action: click[variety pack]
Observation of Best Successor State: You have clicked variety pack.
Reflection of the Best Successor State: The last action and observation of clicking on variety pack does capture one of the required attributes. However, the item is not confirmed to be dairy free and the price is not yet confirmed to be less than $30.00. Therefore, the best successor state captures a step with a low likelihood of leading to purchasing the correct item.
Thus the correctness score is 2.

New Task:
{input}

Respond with the reflection for the last observation of the new task ONLY. As a reminder the last action and observation is as follows:
{last_action}
Your response should start with "Reflection:", include a lookahead step, and end with "Thus the correctness score is ...".'''


score_prompt_feedback = '''Given an item to purchase and a trajectory, analyze the following trajectory, then at the last line conclude "Thus the correctness score is {s}", where s is an integer from 1 to 10.
Here are some examples

Webshop
Instruction:
i am looking for dairy free and apple variety pack of chips, and price lower than 30.00 dollars
[Search]

Action: search[dairy free and apple variety pack of chips]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B07HRFJWP8]
Enjoy Life Foods Soft Baked Ovals, Breakfast Bars, Nut Free Bars, Soy Free, Dairy Free, Non GMO, Gluten Free, Vegan, Variety Pack, 4 Boxes (20 Bars Total)
$100.0
[B01KMHY5PG]
Enjoy Life Soft Baked Chewy Bars, Variety Pack, Nut Free Bars, Soy Free, Dairy Free, Gluten Free, 6 Boxes (30 Total Bars)
$21.49
[B008D2X8C4]
Enjoy Life Lentil Chips Variety Pack, Dairy Free Chips, Soy Free, Nut Free, Non GMO, Vegan, Gluten Free, 24 Bags (0.8 oz)
$100.0

Action: think[The first two items are not what I want. The third item is the dairy free and apple variety pack of chips I want.]
Observation: OK.

Action: click[B008D2X8C4]
Observation:
[Back to Search]
[< Prev]
flavor name [dill and sour cream][garlic & parmesan][light sea salt][margherita pizza][thai chili lime][variety pack]
size [0.8 ounce (pack of 24)][4 ounce (pack of 12)]
Price: $100.0
Rating: N.A.
[Description]
[Features]
[Reviews]
[Buy Now]

Action: think[The item has options 'variety pack' and '0.8 ounce (pack of 24)' and seems good to buy.]
Observation: OK.

Action: click[variety pack]
Observation: You have clicked variety pack.

Action: click[0.8 ounce (pack of 24)]
Observation: You have clicked 0.8 ounce (pack of 24).

Action: click[Buy Now]

Reflection: In this attempt, I was unsuccessful. I accidentally bought a product that was $100, which is more than my budget of $30. Either way, the initial search results were not good. Next time, I will do search["variety pack of chips"] and then check if the results meet the dairy free and the $30 budget constraints. I will continue to refine my searches so that I can find more products.
Thus the correctness score is 5

Webshop 
Instruction:  
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[Search]  

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

Action: think[B078GWRC1J and B078GTKVXY are bright citrus deodorant less then 50 dollars. I can check B078GWRC1J first.]
Observation: OK.
Reflection: Although the task is not yet complete, the first search is correct and returns relevant results. The thought is also logical and has potential to lead to the correct selection.
Thus the correctness score is 9
{trajectories}
{input}
'''