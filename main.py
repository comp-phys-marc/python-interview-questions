import collections
import hashlib
import numpy as np
import pandas as pd

from functools import reduce
from numpy import long


# Sorting


def selection_sort(array):
    """
    The selection sort has a complexity of O(n^2) due to the nested loops.
    :param array: the array of items to sort.
    :return: the sorted array.
    """
    i = 0
    while i < len(array):  # loop 1
        min_element = min(array[i:])  # loop 2 is hidden in the pythonic min function
        del array[array[i:].index(min_element) + i]
        array.insert(i, min_element)
        i += 1

    return array


def bubble_sort(array):
    """
    The bubble sort has an average complexity of O(n^2), or O(n) at best.
    :param array: the array of items to sort.
    :return: the sorted array.
    """
    in_order = False

    while not in_order:
        in_order = True
        i = 0
        while i + 1 < len(array):
            if array[i + 1] < array[i]:
                in_order = False
                temp = array[i]
                array[i] = array[i + 1]
                array[i + 1] = temp
            i += 1

    return array


def insertion_sort(array):
    """
    The insertion sort has an average complexity of O(n^2), or O(n) at best.
    :param array: the array of items to sort.
    :return: the sorted array.
    """
    i = 1

    while i < len(array):
        j = i - 1
        while array[i] < array[j] and j > -1:
            j -= 1
        temp = array[i]
        del array[i]
        array.insert(j + 1, temp)
        i += 1

    return array


def heap_sort(array):
    """
    The heap sort has a complexity of O(n log(n)).
    :param array: the array of items to sort.
    :return: the sorted array.
    """
    heap_size = len(array)

    for i in range(heap_size//2 - 1, -1, -1):
        array = heapify(array, heap_size, i)

    for i in range(heap_size - 1, 0, -1):
        array[i], array[0] = array[0], array[i]
        array = heapify(array, i, 0)

    return array


def heapify(array, heap_size, parent_index=0):
    """
    Transforms a given array into a binary tree representation,
    where the leaves of a node i are given by the values at 2*i+1 and 2*i+2.
    :param array: the array to organize as a heap.
    :return: the 'heapified' array.
    """
    i = parent_index

    largest_index = i
    left_index = 2 * i + 1
    right_index = 2 * i + 2

    if left_index < heap_size:
        left = array[left_index]
        if left > array[largest_index]:
            largest_index = left_index
    if right_index < heap_size:
        right = array[right_index]
        if right > array[largest_index]:
            largest_index = right_index

    if parent_index != largest_index:
        array[parent_index], array[largest_index] = array[largest_index], array[parent_index]

        heapify(array, heap_size, largest_index)

    return array


def quick_sort(array, partition_func, lower_boundary, upper_boundary):
    """
    The quick sort has a complexity of O(n log(n)), or O(n^2) in the worst case.
    :param array: the array of items to sort.
    :return: the sorted array.
    """
    if lower_boundary < upper_boundary:  # terminate on partition boundaries' convergence
        quick_sort(array, partition_func, lower_boundary, partition_func(array, lower_boundary, upper_boundary) - 1)  # sort the lower partition
        quick_sort(array, partition_func, partition_func(array, lower_boundary, upper_boundary) + 1, upper_boundary)  # sort the upper partition

    return array


def quick_sort_partition_func(array, lower_boundary, upper_boundary):
    """
    The partition function that recursively sorts the array given to quick_sort.
    :param array: the array to sort.
    :param lower_boundary: the lower boundary of the current partition.
    :param upper_boundary: the upper boundary of the current partition.
    :return:
    """
    pivot = array[upper_boundary]
    i = lower_boundary - 1  # start with the element preceding the lower_boundary
    j = lower_boundary

    while j <= (upper_boundary - 1):  # close the distance between the lower and upper boundaries
        if array[j] < pivot:
            i += 1  # continue to the next ith element

            temp = array[i]  # swap the ith and jth elements, moving the lower values to the right
            array[i] = array[j]
            array[j] = temp

        j += 1

    temp = array[i + 1]  # swap the upper boundary and the new pivot's values
    array[i + 1] = array[upper_boundary]
    array[upper_boundary] = temp

    return i + 1  # return the new pivot


def merge_sort(array):
    """
    The merge sort has a complexity of O(n log(n)).
    :param array: the array of items to sort.
    :return: the sorted array.
    """

    if len(array) <= 1:
        return array

    middle = len(array) // 2

    left = array[:middle]
    right = array[middle:]

    merge_sort(left)
    merge_sort(right)

    i = 0
    j = 0
    k = 0

    # put elements back in array in order
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            array[k] = left[i]
            i += 1
        else:
            array[k] = right[j]
            j += 1
        k += 1

    # In case of an odd length array
    while i < len(left):
        array[k] = left[i]
        i += 1
        k += 1

    while j < len(right):
        array[k] = right[j]
        j += 1
        k += 1

    return array


# Given five positive integers, find the minimum and maximum values that can be calculated by summing exactly
# four of the five integers. Then print the respective minimum and maximum values as a single line of two
# space-separated long integers.


def mini_max_sum(arr):
    if len(arr) != 5:
        print('please provide an array of 5 integers')

    else:
        sorted = quick_sort_five(arr)

        max = reduce(lambda a, b: a + b, sorted[:4])
        min = reduce(lambda a, b: a + b, sorted[1:])

        print('{0} {1}'.format(long(max), long(min)))


def quick_sort_five(arr):
    pivot = 0  # choose a pivot at the center of the list

    while pivot < len(arr):
        pivot += 1
        done = False

        while not done:
            i = 0
            j = len(arr) - 1

            while i <= pivot + 1:
                while j >= pivot:

                    if arr[i] < arr[j]:
                        temp = arr[i]
                        arr[i] = arr[j]
                        arr[j] = temp

                        i = 0
                        j = len(arr) - 1

                    elif i == j:
                        done = True
                        break

                    elif j == pivot:
                        break

                    elif j - 1 >= pivot:
                        j -= 1

                i += 1

                if i == j:
                    done = True
                    break

    return arr


# Encryption


def encrypt_string(hash_string):
    sha_signature = \
        hashlib.sha256(hash_string.encode()).hexdigest()
    return sha_signature


# Add two numbers without using arithmetic operators


def add_bit_method(first, second):

    if first == 0:
        return second

    while (second != 0):  # does not enter if second number is 0

        carry = first & second  # carry given by indexes where binary representations of numbers both have 1s
        first = first ^ second  # summing performed by union operator (a 0 meets a 1 -> 1)
        second = carry << 1  # the carry is 'carried' by a 1 index shift for the next round

    return first


def add_hex_method(first, second):

    if first == '0':
        return second
    elif second == '0':
        return first

    i = 0
    j = 0

    sum = ''
    sum_value = 0

    while i < len(first) and j < len(second):
        first_value = int(hex(ord(first[i])), 16)  # string -> unicode -> hex
        second_value = int(hex(ord(second[j])), 16)  # string -> unicode -> hex
        sum_value += (first_value + second_value) * (10 ** i)  # add value to sum
        i += 1
        j += 1

    # in case of unequal lengths

    while i < len(first):
        sum_value += int(hex(ord(first[i])), 16) * (10 ** i)
        i += 1

    while j < len(first):
        sum_value += int(hex(ord(second[j])), 16) * (10 ** j)
        j += 1

    sum += bytes.fromhex(hex(sum_value)).decode('utf-8')  # concatenate to the sum string

    return first


# Given an integer, return the integer with reversed digits.
# Note: The integer could be either positive or negative.


def reverse_integer(integer):
    reversed = []
    temp = str(abs(integer))  # remove sign, convert to string

    for char in temp:
        reversed.insert(0, char)  # put in array in reverse order

    if integer < 0:  # prepend sign if necessary
        reversed.insert(0, "-")

    return int("".join(reversed))  # join reversed elements, return as int


def reverse_integer_slicing(integer):
    string = str(integer)  # convert to string

    if string[0] == "-":
        return int("-" + string[:0:-1])  # return entire string in reverse order, as a negative int
    else:
        return int(string[::-1])  # return entire string in reverse order, as an int


# For a given sentence, return the average word length.
# Note: Remember to remove punctuation first.


def average_word_length(sentence):
    words = sentence\
        .replace(".", "")\
        .replace("?", "")\
        .replace("!", "")\
        .replace(",", "")\
        .split(" ")  # remove special characters and split along spaces into words

    size_sum = reduce(lambda size_sum, word: size_sum + word,  # add the lengths together
                     map(lambda word: len(word), words))  # get the length of each word

    average = size_sum / len(words)  # avg = total / count

    return average


# Given two non-negative integers num1 and num2 represented as string, return the sum of num1 and num2.
# You must not use any built-in BigInteger library or convert the inputs to integer directly.


def average_word_length_list_comprehension(sentence):
    for p in "!?',;.":  # remove special chars
        sentence = sentence.replace(p, '')
    words = sentence.split()  # split into words
    return round(sum(len(word) for word in words) / len(words), 2)  # calculate and return avg


def add_strings(string1, string2):
    return str(eval(string1), eval(string2))  # use eval to "unwrap" the numbers so they ar eno linger strings


def add_strings_helper(num):  # converts to a number using ord
    magnitude = 10 ** (len(num) - 1)
    result = 0
    for i in num:
        result += (ord(i) - ord("0")) * magnitude  # calculate the offset of each digit from "0" bytecode,
        magnitude = magnitude//10                  # and multiply this by its order of magnitude
    return result


def add_strings_explicit(num1, num2):
    return str(add_strings_helper(num1) + add_strings_helper(num2))


# Given a string, find the first non-repeating character in it and return its index.
# If it doesn't exist, return -1. # Note: all the input strings are already lowercase.


def first_unique_char(string):
    freq = {}  # freq dict
    indeces = {}  #ideces dict

    index = 1
    for char in string:
        if char in freq.keys():
            freq[char] += 1  # increment freq
        else:
            indeces[char] = index  # record first instance's index
            freq[char] = 1
        index += 1

    unique_indeces = list(filter(lambda index: index != 0,  # filter out non-unique entries
                                 map(lambda char: (freq[char] == 1) * indeces[char], freq.keys())))  # return indeces of unique nums

    return min(unique_indeces) - 1  # correct index


def first_unique_char_collections(string):
    count = collections.Counter(string)  # creates freq dict

    for index, char in enumerate(string):  # check chars in order
        if count[char] == 1:
                return index  # return index of first char with one occurence

    return -1


# Given a non-empty string s, you may delete at most one character. Judge whether you can make it a palindrome.
# The string will only contain lowercase characters a-z.


def is_palindrome(string):  # check if string is same forward as backward
    start = 0
    end = len(string) - 1

    while start < end:  # stop at the middle
        if string[start] != string[end]:
            return False
        start += 1
        end -= 1

    return True


def valid_palindrome(string):
    possible_removals = [string]  # no removals is an option

    for char in range(len(string)):  # remove each char one at a time
        possible_removals.append(string[: char] + string[char + 1:])

    for possibility in possible_removals:  # check if each possibility is a palindrome
        if is_palindrome(possibility):
            return True

    return False


def valid_palindrome_slicing(string):
    for i in range(len(string)):
        possibility = string[:i] + string[i + 1:]  # remove each char one at a time
        if possibility == possibility[::-1]:  # check if each possibility is a palindrome
            return True

    return possibility == possibility[::-1]  # no removals is an option


# Given an array of integers, determine whether the array is monotonic or not.


def monotonic_array(array):
    last = array[0]
    next = array[1]

    direction = next > last  # check monotomic direction

    for i in range(1, len(array)):
        next = array[i]
        if (direction and (array[i] >= last))\
                or ((not direction) and (array[i] <= last)):  # check that direction is preserved
            last = next
            i += 1
        else:
            return False

    return True


def monotonic_array_using_comprehension(nums):
    return (all(nums[i] <= nums[i + 1] for i in range(len(nums) - 1)) or
            all(nums[i] >= nums[i + 1] for i in range(len(nums) - 1)))


#Given an array nums, write a function to move all zeroes to the end of it while maintaining the relative order of
#the non-zero elements.


def move_zeroes(numbers):
    for number in numbers:
        if number == 0:
            numbers.remove(number)  # remove each found zero
            numbers.append(0)  # add one zero to the end per found zero
    return numbers


# Given an array containing None values fill in the None values with most recent
# non None value in the array


def fill_the_blanks(array):
    for i in range(len(array)):
        if array[i] is None and i > 0:  # cannot fill in None at index zero
            array[i] = array[i - 1]  # carry last non-None val forward
        i += 1
    return array


#Given two sentences, return an array that has the words that appear in one sentence and not
#the other and an array with the words in common.


def matched_and_mismatched_words_helper(array1, array2):
    matched = []
    mismatched = []

    for word in array1:  # check for word's membership in other array
        if word in array2 and word not in matched:
            matched.append(word)
        elif word not in array2 and word not in mismatched:
            mismatched.append(word)

    return matched, mismatched


def matched_and_mismatched_words(array1, array2):
    matched, mismatched = matched_and_mismatched_words_helper(array1, array2)  # check each word in array1
    matched_reversed, mismatched_reversed = matched_and_mismatched_words_helper(array2, array1)  # check each word in array2
    return set(matched + matched_reversed), set(mismatched + mismatched_reversed)  # return unique sets


def matched_and_mismatched_words_using_set_ops(array1, array2):
    set1 = set(array1)
    set2 = set(array2)
    return sorted(list(set1^set2)), sorted(list(set1&set2))  # symmetric difference and intersection


# Given k numbers which are less than n, return the set of prime number among them


def prime_nums(domain):
    prime_nums = []
    for num in range(2, domain):
        prime = True
        for i in range(2, num):  # iterate over possible divisors
            if (num % i) == 0:  # check if divisible
                prime = False
                break
        if prime:
            prime_nums.append(num)  # if has no divisors > 1, it is prime
    return prime_nums


# A function that prints the numbers ranging from one to 50, but for multiples of three,
# prints "Fizz" instead of the number, and for the multiples of five, prints "Buzz." and
# for numbers which are multiples of both three and five, prints "FizzBuzz"


def fizz_buzz():
    for i in range(1, 51):
        if (i % 3 != 0) and (i % 5 != 0):
            print(i, end="\n")
            continue
        if i % 3 == 0:
            print("Fizz", end="")
        if i % 5 == 0:
            print("Buzz", end="")
        print("\n")


# Given a function that can produce an image/array from the top-left pixel coordinates
# of the original image, tile the image


def tile_image(image, window, steps=None):
    """
    Tiles an image using a striding method. The striding method makes it possible
    to process images that are not symmetrical. For symmetrical images, sliding_window_view
    can be used instead of as_strided, which doesn't accept a strides tuple.

    :param image: The image to tile.
    :param window: The dimension of the window (tile) as a tuple.
    :param steps: The steps (number of strides) in each dimension as a tuple.
    :return:
    """
    image_shape = np.array(image.shape)  # gives an array with the shape of the image
    window_shape = np.array(window).reshape(-1)  # gives an array with a shape inferred from the window

    if steps:  # if steps are provided...
        step = np.array(steps).reshape(-1)  # gives an array with a shape inferred from the steps
    else:  # if steps are not provided...
        step = np.ones_like(image_shape)  # gives an array of ones with the shape of the image

    # strides give the number of memory locations needed to scan in order
    # to reach the next pixel in a row and in a column, respectively.
    image_strides = np.array(image.strides)

    assert np.all(np.r_[  # np.r_ performs a row-wise array merge to prepare the conditions array for np.all
        image_shape.size == window_shape.size,
        window_shape.size == step.size,
        window_shape <= image_shape
    ])

    shape = tuple((image_shape - window_shape) // step + 1) \
            + tuple(window_shape)  # the output shape will be sufficient to fit the total number of tiles
    strides = tuple(image_strides * step) + tuple(image_strides)  # we advance one stride per step
    as_strided = np.lib.stride_tricks.as_strided

    # as_strided creates a view into the array with the given shape and strides
    image_view = as_strided(image, shape=shape, strides=strides)

    # a view is returned, which has the advantage of referencing the
    # image's data in memory rather than copying it
    return image_view


# Fill in missing values in a given dataset using the mean value from the available data


def handle_missing_values(data_array, axis='index'):
    data_frame = pd.DataFrame(
        data=data_array[1:, 1:],
        index=data_array[1:, 0],
        columns=data_array[0, 1:]
    )

    means = data_frame.mean(
        axis=1,
        skipna=True,
        numeric_only=True
    )
    return data_frame.fillna(value=means, axis=axis)


# Implement a binary search tree


class BinaryTree:

    def __init__(self, root=None):
        self.root = root

    def insert_node(self, node):
        return self.root.insert_node(node)

    def insert_value(self, value):
        return self.root.insert_value(value)

    def find_parent(self, node):
        if node == self.root:
            return None

        return self._find_parent_helper(self.root, node)

    def _find_parent_helper(self, current_node, node):
        if current_node.left == node or current_node.right == node:
            return current_node
        elif node < current_node:
            return self._find_parent_helper(current_node.left, node)
        elif node > current_node:
            return self._find_parent_helper(current_node.right, node)

    def print_tree(self):
        self._print_tree_helper(self.root, space=0)

    def _print_tree_helper(self, node, space):
        if node is None:
            return

        self._print_tree_helper(node.left, space + 10)
        print("\n", end="")

        for i in range(space):
            print(" ", end="")

        print(node.value, end="")
        print("\n", end="")

        self._print_tree_helper(node.right, space + 10)


class Node:

    def __init__(self, value=None):
        self.parent = None
        self.left = None
        self.right = None
        self.value = value

    def __str__(self):
        return str(self.value)

    def __eq__(self, other_node):
        return self.value == other_node.value

    def __lt__(self, other_node):
        return self.value < other_node.value

    def __gt__(self, other_node):
        return self.value > other_node.value

    def insert_value(self, value=None):
        return self.insert_node(Node(value))

    def insert_node(self, node):
        if node == self:
            return self  # no need to insert the value - it is already in the tree
        elif node > self:
            if self.right is not None:
                return self.right.insert_node(node)
            else:
                self.right = node
        else:  # the value is less than that of the current node
            if self.left is not None:
                return self.left.insert_node(node)
            else:
                self.left = node

    def remove(self, binary_tree):
        parent = binary_tree.find_parent(self)

        left_node = self.left
        right_node = self.right

        if parent.left == self:
            parent.left = None
        elif parent.right == self:
            parent.right = None

        parent.insert_node(left_node)
        parent.insert_node(right_node)

        del self  # remove self

        return parent

    def reverse_tree(self):
        left_node = self.left
        right_node = self.right

        self.left = right_node
        self.right = left_node

        self.left.reverse_tree()
        self.right.reverse_tree()

        return self

    def find_node(self, value):
        if self.value == value:
            return self
        elif self.value < value:
            if self.left is not None:
                return self.left.find_node(value)
            else:
                return None
        elif self.value > value:
            if self.right is not None:
                return self.right.find_node(value)
            else:
                return None


# Preorder binary tree traversal


def pre_order_helper(root):
    st = ''
    st += str(root)

    if root.left is not None:
        st += ' ' + pre_order_helper(root.left)

    if root.right is not None:
        st += ' ' + pre_order_helper(root.right)

    return st


def pre_order(root):
    st = ''
    st += pre_order_helper(root)

    print(st)


# Miscellaneous


# Given a time in -hour AM/PM format, convert it to military (24-hour) time.


def time_conversion(s):
    am = (s[len(s) - 2:] == 'AM')
    time = s[:len(s) - 2]

    split = time.split(':')

    hour = int(split[0])
    minute = int(split[1])
    second = int(split[2])

    if am and hour == 12:
        hour = 0

    elif not am and hour != 12:
        hour += 12

    print(f'{str(hour).zfill(2)}:{str(minute).zfill(2)}:{str(second).zfill(2)}')


# Given an array of integers, calculate the ratios of its elements that are positive, negative, and zero.
# Print the decimal value of each fraction on a new line with places after the decimal.


def plus_minus(arr):
    size = len(arr)
    pos = 0
    neg = 0
    if size <= 100:
        for el in arr:
            if -100 > el or el > 100:
                print('Array elements must be values between -100 and 100')
                break
            elif el > 0:
                pos += 1
            elif el < 0:
                neg += 1

        zeroes = size - pos - neg

        print('{0:.6f}'.format(pos / size))
        print('{0:.6f}'.format(neg / size))
        print('{0:.6f}'.format(zeroes / size))

    else:
        print('Provided array must be of length <= 100')


if __name__ == '__main__':
    
    bin_tree = BinaryTree(Node(5))
    root = bin_tree.root
    bin_tree.insert_node(Node(3))
    root.left.insert_value(2)
    root.insert_value(7)
    root.insert_value(6)
    root.insert_value(8)
    root.insert_value(4)
    bin_tree.print_tree()
    
    print(selection_sort([4, 7, 1, 2, 7, 0, 9, 3, 1, 2, 5, 7]))
    print(bubble_sort([4, 7, 1, 2, 7, 0, 9, 3, 1, 2, 5, 7]))
    print(insertion_sort([4, 7, 1, 2, 7, 0, 9, 3, 1, 2, 5, 7]))
    print(heap_sort([4, 7, 1, 2, 7, 0, 9, 3, 1, 2, 5, 7]))
    print(
        quick_sort(
            [4, 7, 1, 2, 7, 0, 9, 3, 1, 2, 5, 7],
            quick_sort_partition_func,
            0,
            len([4, 7, 1, 2, 7, 0, 9, 3, 1, 2, 5, 7]) - 1
        )
    )
    print(merge_sort([4, 7, 1, 2, 7, 0, 9, 3, 1, 2, 5, 7]))

    print(add_bit_method(5, 6))
    print(add_hex_method('5', '6'))
    plus_minus([1, 1, 0, -1, -1])
    mini_max_sum([7, 69, 2, 221, 8974])

    for t in ['AM', 'PM']:
        for hour in range(1, 13):
            for minute in range(1):
                for second in range(1):
                    time_conversion(f'{str(hour).zfill(2)}:{str(minute).zfill(2)}:{str(second).zfill(2)}{t}')

    time_conversion('12:45:54PM')
    pre_order(root)
