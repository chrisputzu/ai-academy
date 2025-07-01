from typing import List

# Codice sporco da pulire

# Translated into English to use an unique and coherent universal language
def create_list(list_n1: List[int], list_n2: List[int], n3: int) -> List[int]:
    """
    Performs conditional operations on two lists of integers and a number constant
    then find Large values in n > 50, zeros or other values.

    Parameters
    ----------
        list_n1 : (List[int]) 
        First list of integers.
        list_n2 : (List[int])
        Second list of integers.
        n3 : (int)
        Constant integer value to be added under certain conditions.

    Returns
    -------
        list_int : List[int]: 
        Returns a manipulated list of integers.
    """
    list_int = []

    for n1, n2 in zip(list_n1, list_n2):
        if n1 > 10:
            list_int.append(n1 + n2 + n3)
        elif n2 > 5:
            list_int.append(n1 * 2 + n2)
        else:
            list_int.append(0)
    
    return list_int

def check_largevalues_or_zeros(list_int: List[int]) -> None: 
    """
    Performs a check to find large values or zeros for integers greater than 50.

    Parameters
    ----------

    list_int : List[int]
    Input list of manipulated integers.

    Returns
    -------
    None 
    """
    for el in list_int:

        if el > 50:
            print("Large value found!")
        elif el == 0:
            print("Zero found!")
        else:
            print("Value:", el)

def sum_positive_integers(n1: int, n2: int, n3: int, n4: int) -> int:
    """ 
    Performs a Check where all numbers are positive and 
    returns their sum.

    Parameters
    ----------
        n1 : (int) 
        First random integer.
        n2 : (int)
        Second random integer.
        n3 : (int)
        Third random integer.
        n4 : (int)
        Fourth random integer.

    Returns
    -------
        pos_integers_sum : (int)
        Sum of positive integers.
    """
    if n1 > 0 and n2 > 0 and n3 > 0:
        print("All positive")
        pos_integers_sum = n1 + n2 + n3 + n4

    return pos_integers_sum

def final_counter(list_int: List[int]) -> int:
    """
    Performs an analysis to determine if integers are positive or negative
    then add o subtract from the final count.

    Parameters
    ----------
        list_int : (List[int]) 
        List of integers.

    Returns
    -------
        final_count : int: 
        Processed integer. 
    """
    final_count = 0
    for el in list_int:
        print("Element:", el)

        if el > 0:
            final_count += el
        else:
            final_count -= el

        return final_count

def check_even_or_odd(final_count: int) -> None:
    """
    Analyzes the final_count integer number to determine if it is even or odd,
    or it is very large.

    Parameters
    ----------
        final_count : (int) 
        Input final count value.

    Returns
    -------
    None 
    """
    if final_count % 2 == 0:
        print("Even:", final_count)
    else:
        print("Odd:", final_count)

    if final_count > 100:
        print("Very large:", final_count)


def main():
    list_n1 = [12, 3, 7, 15]
    list_n2 = [6, 2, 10, 4]
    n3 = 5

    print("\nStart of program!")

    print("\n1. Create list!")
    list_int = create_list(list_n1=list_n1, list_n2=list_n2, n3=n3)

    print("\n2. Check for large values or zeros!")
    check_largevalues_or_zeros(list_int=list_int)

    print("\n3. Sum of int numbers!")
    pos_integers_sum = sum_positive_integers(n1=1, n2=2, n3=3, n4=4)
    print("Sum of positive integers:", pos_integers_sum)

    print("\n4. Final counter!")
    final_count = final_counter(list_int=list_int)
    print("Final count:", final_count)

    print("\n5. Check even or odd!")
    check_even_or_odd(final_count=final_count)

    print("\nEnd of program!")

if __name__ == "__main__":
    main()
