from typing import List

# Codice sporco da pulire

# Translated into English to use an unique and coherent universal language
def process_lists(list_1: List[int], list_2: List[int], num3: int) -> List[int]:
    """
    Performs conditional operations on two lists of integers and a number constant
    then find Large values in n > 50, zeros or other values.
    Parameters
    ----------
        list_1 : (List[int]) 
        First list of integers.
        list_2 : (List[int])
        Second list of integers.
        num3 : (int)
        Constant integer value to be added under certain conditions.
    Returns
    -------
        result : List[int]: 
        Returns a manipulated list of integers.
    """
    result = []

    for num1, num2 in zip(list_1, list_2):
        if num1 > 10:
            result.append(num1 + num2 + num3)
        elif num2 > 5:
            result.append(num1 * 2 + num2)
        else:
            result.append(0)

    return result

def print_value_infos(result: List[int]) -> None: 
    """
    Print info about each value: if it's large (>50), zero, or its value.
    Parameters
    ----------
    result : List[int]
    Input list of manipulated integers.
    Returns
    -------
    None 
    """
    for el in result:

        if el > 50:
            print("Large value found!")
        elif el == 0:
            print("Zero found!")
        else:
            print("Value:", el)

def sum_if_all_positive(num1: int, num2: int, num3: int, num4: int) -> int:
    """ 
    Performs a Check where all numbers are positive and 
    returns their sum.
    Parameters
    ----------
        num1 : (int) 
        First random integer.
        num2 : (int)
        Second random integer.
        num3 : (int)
        Third random integer.
        num4 : (int)
        Fourth random integer.
    Returns
    -------
        pos_integers_sum : (int)
        Sum of positive integers.
    """
    if num1 > 0 and num2 > 0 and num3 > 0:
        print("All positive")
        pos_integers_sum = num1 + num2 + num3 + num4

    return pos_integers_sum

def aggregate_count(result: List[int]) -> int:
    """
    Add positives, subtract negatives/zeros from a running total.
    Parameters
    ----------
        result : (List[int]) 
        List of integers.
    Returns
    -------
        total : int: 
        Processed integer. 
    """
    total = 0
    for el in result:
        print("Element:", el)

        if el > 0:
            total += el
        else:
            total -= el

        return total

def print_even_odd_info(total: int) -> None:
    """
    Print if n is even or odd and if it's very large (>100).
    Parameters
    ----------
        total : (int) 
        Input final count value.
    Returns
    -------
    None 
    """
    if total % 2 == 0:
        print("Even:", total)
    else:
        print("Odd:", total)

    if total > 100:
        print("Very large:", total)


def main():
    list_1 = [12, 3, 7, 15]
    list_2 = [6, 2, 10, 4]
    num3 = 5

    result = process_lists(list_1=list_1, list_2=list_2, num3=num3)
    print_value_infos(result=result)

    pos_integers_sum = sum_if_all_positive(num1=1, num2=2, num3=3, num4=4)
    print("Sum of positive integers:", pos_integers_sum)

    total = aggregate_count(result=result)
    print("Final count:", total)

    print_even_odd_info(total=total)





if __name__ == "__main__":
    main()




