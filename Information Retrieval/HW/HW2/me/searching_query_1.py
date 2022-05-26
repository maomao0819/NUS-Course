def Tokenize(Query):
    # split with space
    QueryList = Query.split()
    NewQueryList = []
    # split with bracket
    for i in range(len(QueryList)):
        current_token = QueryList.pop(0)
        # left bracket
        if current_token[0] == '(':
            NewQueryList.append('(')
            NewQueryList.append(current_token[1:])
        # right bracket
        elif current_token[-1] == ')':
            NewQueryList.append(current_token[:-1])
            NewQueryList.append(')')
        else:
            NewQueryList.append(current_token)
    return NewQueryList

operators_info  = {
    # operator: (precedence, associativity)
    'NOT': (2, 'L'), 
    'AND': (1, 'L'),
    'OR': (0, 'L')
}

def ShuntYard(tokens):
    tokens += ['end']
    operators = []
    output = []

    while len(tokens) != 1:
        current_token = tokens.pop(0)
        # Is a operator
        if current_token in operators_info.keys():
            while True:
                if len(operators) == 0:
                    break
                satisfied = False
                if operators[-1] not in ["(", ")"]:
                    # operator at top has greater precedence
                    if operators_info[operators[-1]][0] > operators_info[current_token][0]:
                        satisfied = True
                    # equal precedence and has left associativity
                    elif operators_info[operators[-1]][0] == operators_info[current_token][0]:
                        if operators_info[operators[-1]][1] == 'L':
                            satisfied = True
                satisfied = satisfied and operators[-1] != "("
                if not satisfied:
                    break
                output.append(operators.pop())
            operators.append(current_token)
        # Is a left bracket
        elif current_token == "(":
            operators.append(current_token)
        # Is a right bracket
        elif current_token == ")":
            while True:
                if len(operators) == 0:
                    break
                if operators[-1] == "(":
                    break
                output.append(operators.pop())
            if len(operators) != 0 and operators[-1] == "(":
                operators.pop()
        # Not a operator
        else:
            output.append(current_token) 
    output.extend(operators[::-1])
    return output

def ANDqueries(list1, list2):
    # return list(set(list1) & set(list2))
    id1 = 0
    id2 = 0
    output = []
    while id1 < len(list1) and id2 < len(list2):
        if list1[id1] == list2[id2]:
            output.append(list1[id1])
            id1 += 1
            id2 += 1
        elif list1[id1] < list2[id2]:
            id1 += 1
        elif list1[id1] > list2[id2]:
            id2 += 1
    return output

# list1 and not list2 
def ANDNOTqueries(list1, list2):
    id1 = 0
    id2 = 0
    output = []
    while id1 < len(list1) and id2 < len(list2):
        if list1[id1] == list2[id2]:
            id1 += 1
            id2 += 1
        elif list1[id1] < list2[id2]:
            output.append(list1[id1])
            id1 += 1
        elif list1[id1] > list2[id2]:
            id2 += 1
    if id1 < len(list1):
        output += list1[id1:]
    return output

def ORqueries(list1, list2):
    # return list(set(list1) | set(list2))
    id1 = 0
    id2 = 0
    output = []
    while id1 < len(list1) and id2 < len(list2):
        if list1[id1] == list2[id2]:
            output.append(list1[id1])
            id1 += 1
            id2 += 1
        elif list1[id1] < list2[id2]:
            output.append(list1[id1])
            id1 += 1
        elif list1[id1] > list2[id2]:
            output.append(list2[id2])
            id2 += 1
    if id1 < len(list1):
        output += list1[id1:]
    if id2 < len(list2):
        output += list2[id2:]
    return output
    
def searching(Query):
    output = []
    is_former_not = False   # not A and B 
    is_later_not = False    # A and not B 
    while len(Query):
        current_token = Query.pop(0)
        # Is a operator
        if current_token in operators_info.keys():
            if len(output) < 2:
                print('The query has an error!')
            else:
                # NOT
                if current_token == 'NOT':
                    if Query[0] in operators_info.keys():
                        is_later_not = True
                    else:
                        is_former_not = True
                # AND, OR
                else:
                    list2 = output.pop(-1)
                    list1 = output.pop(-1)
                    if current_token == 'AND':
                        if is_former_not:
                            output.append(ANDNOTqueries(list2, list1))
                            is_former_not = False
                        elif is_later_not:
                            output.append(ANDNOTqueries(list1, list2))
                            is_later_not = False 
                        else:
                            output.append(ANDqueries(list1, list2))
                    elif current_token == 'OR':
                        output.append(ORqueries(list1, list2))
        # Not a operator
        else:
            output.append(dictionary[current_token])
    return output[0]

dictionary = {
    'bill': [1, 2, 3, 4, 5],
    'Gates': [2, 3, 4, 5, 6],
    'vista': [1, 7, 9],
    'XP': [2, 5, 6, 8],
    'mac': [1, 2, 3]
}

Query = 'bill OR Gates AND (vista OR XP) AND NOT mac'
print('Initial query', Query)
# Query = 'bill OR Gates AND (vista OR XP) AND mac'
QueryList = Tokenize(Query)
print('Before Shunt Yard', QueryList)
QueryList = ShuntYard(QueryList)
print('After Shunt Yard', QueryList)
print('searching result', searching(QueryList))