/**
 * @file mm.c
 * @brief A 64-bit struct-based segregated free list memory allocator
 *
 * 15-213: Introduction to Computer Systems
 *
 * This program implements a dynamic memory allocator using segregated
 * free list, LIFO, and first fit.
 *
 *************************************************************************
 *
 * ADVICE FOR STUDENTS.
 * - Step 0: Please read the writeup!
 * - Step 1: Write your heap checker.
 * - Step 2: Write contracts / debugging assert statements.
 * - Good luck, and have fun!
 *
 *************************************************************************
 *
 * @author Yujia Wang <yujiawan@andrew.cmu.edu>
 */

#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "memlib.h"
#include "mm.h"

/* Do not change the following! */

#ifdef DRIVER
/* create aliases for driver tests */
#define malloc mm_malloc
#define free mm_free
#define realloc mm_realloc
#define calloc mm_calloc
#define memset mem_memset
#define memcpy mem_memcpy
#endif /* def DRIVER */

/* You can change anything from here onward */

/*
 *****************************************************************************
 * If DEBUG is defined (such as when running mdriver-dbg), these macros      *
 * are enabled. You can use them to print debugging output and to check      *
 * contracts only in debug mode.                                             *
 *                                                                           *
 * Only debugging macros with names beginning "dbg_" are allowed.            *
 * You may not define any other macros having arguments.                     *
 *****************************************************************************
 */
#ifdef DEBUG
/* When DEBUG is defined, these form aliases to useful functions */
#define dbg_printf(...) printf(__VA_ARGS__)
#define dbg_requires(expr) assert(expr)
#define dbg_assert(expr) assert(expr)
#define dbg_ensures(expr) assert(expr)
#define dbg_printheap(...) print_heap(__VA_ARGS__)
#else
/* When DEBUG is not defined, no code gets generated for these */
/* The sizeof() hack is used to avoid "unused variable" warnings */
#define dbg_printf(...) (sizeof(__VA_ARGS__), -1)
#define dbg_requires(expr) (sizeof(expr), 1)
#define dbg_assert(expr) (sizeof(expr), 1)
#define dbg_ensures(expr) (sizeof(expr), 1)
#define dbg_printheap(...) ((void)sizeof(__VA_ARGS__))
#endif

/* Basic constants */
typedef uint64_t word_t;

/** @brief Word and header size (bytes) */
static const size_t wsize = sizeof(word_t);

/** @brief Double word size (bytes) */
static const size_t dsize = 2 * wsize;

/** @brief Minimum block size (bytes) */
static const size_t min_block_size = 2 * dsize;

/**
 * @brief Minimum size (bytes) when extending heap
 * (Must be divisible by dsize)
 */
static const size_t chunksize = (1 << 13);

/**
 * @brief Extract allocated bit in header/footer of block
 */
static const word_t alloc_mask = 0x1;

/**
 * @brief Extract allocated bit of previous block in header/footer of block
 */
static const word_t prev_alloc_mask = 0x2;

/**
 * @brief Extract block size
 */
static const word_t size_mask = ~(word_t)0xF;

/** @brief Number of seglist size classes */
static const size_t seglist_max = 10;

/** @brief Represents the header and payload of one block in the heap */
typedef struct block {
    /** @brief Header contains size + allocation flag */
    word_t header;

    /**
     * @brief Union represents next and previous pointers in free blocks,
     * and payload in allocated blocks.
     */
    union {
        struct {
            /** @brief A pointer to next block. */
            struct block *next;
            /** @brief A pointer to previous block. */
            struct block *prev;
        };
        /** @brief A pointer to payload memory. */
        char payload[0];
    };
} block_t;

/* Global variables */
/** @brief Array of pointers to seglists of differernt size classes. */
static block_t *seglist[seglist_max];

/** @brief Pointer to first block in the heap */
static block_t *heap_start = NULL;

/*
 * ---------------------------------------------------------------------------
 *                        BEGIN SHORT HELPER FUNCTIONS
 * ---------------------------------------------------------------------------
 */

/**
 * @brief Returns the maximum of two integers.
 * @param[in] x
 * @param[in] y
 * @return `x` if `x > y`, and `y` otherwise.
 */
static size_t max(size_t x, size_t y) {
    return (x > y) ? x : y;
}

/**
 * @brief Rounds `size` up to next multiple of n
 * @param[in] size
 * @param[in] n
 * @return The size after rounding up
 */
static size_t round_up(size_t size, size_t n) {
    return n * ((size + (n - 1)) / n);
}

/**
 * @brief Packs the `size` and `alloc` of a block into a word suitable for
 *        use as a packed value.
 *
 * Packed values are used for both headers and footers.
 *
 * The allocation status is packed into the lowest bit of the word.
 *
 * @param[in] size The size of the block being represented
 * @param[in] alloc True if the block is allocated
 * @return The packed value
 */
static word_t pack(size_t size, bool alloc, bool prev_alloc) {
    word_t word = size;
    if (alloc) {
        word |= alloc_mask;
    }
    if (prev_alloc) {
        word |= prev_alloc_mask;
    }
    return word;
}

/**
 * @brief Extracts the size represented in a packed word.
 *
 * This function simply clears the lowest 4 bits of the word, as the heap
 * is 16-byte aligned.
 *
 * @param[in] word
 * @return The size of the block represented by the word
 */
static size_t extract_size(word_t word) {
    return (word & size_mask);
}

/**
 * @brief Extracts the size of a block from its header.
 * @param[in] block
 * @return The size of the block
 */
static size_t get_size(block_t *block) {
    return extract_size(block->header);
}

/**
 * @brief Given a payload pointer, returns a pointer to the corresponding
 *        block.
 * @param[in] bp A pointer to a block's payload
 * @return The corresponding block
 */
static block_t *payload_to_header(void *bp) {
    return (block_t *)((char *)bp - offsetof(block_t, payload));
}

/**
 * @brief Given a block pointer, returns a pointer to the corresponding
 *        payload.
 * @param[in] block
 * @return A pointer to the block's payload
 * @pre The block must be a valid block, not a boundary tag.
 */
static void *header_to_payload(block_t *block) {
    dbg_requires(get_size(block) != 0);
    return (void *)(block->payload);
}

/**
 * @brief Given a block pointer, returns a pointer to the corresponding
 *        footer.
 * @param[in] block
 * @return A pointer to the block's footer
 * @pre The block must be a valid block, not a boundary tag.
 */
static word_t *header_to_footer(block_t *block) {
    dbg_requires(get_size(block) != 0 &&
                 "Called header_to_footer on the epilogue block");
    return (word_t *)(block->payload + get_size(block) - dsize);
}

/**
 * @brief Given a block footer, returns a pointer to the corresponding
 *        header.
 * @param[in] footer A pointer to the block's footer
 * @return A pointer to the start of the block
 * @pre The footer must be the footer of a valid block, not a boundary tag.
 */
static block_t *footer_to_header(word_t *footer) {
    size_t size = extract_size(*footer);
    dbg_assert(size != 0 && "Called footer_to_header on the prologue block");
    return (block_t *)((char *)footer + wsize - size);
}

/**
 * @brief Returns the payload size of a given block.
 *
 * The payload size is equal to the entire block size minus the sizes of the
 * block's header and footer.
 *
 * @param[in] block
 * @return The size of the block's payload
 */
static size_t get_payload_size(block_t *block) {
    size_t asize = get_size(block);
    return asize - wsize;
}

/**
 * @brief Returns the allocation status of a given header value.
 *
 * This is based on the lowest bit of the header value.
 *
 * @param[in] word
 * @return The allocation status correpsonding to the word
 */
static bool extract_alloc(word_t word) {
    return (bool)(word & alloc_mask);
}

/**
 * @brief Returns the allocation status of a block, based on its header.
 * @param[in] block
 * @return The allocation status of the block
 */
static bool get_alloc(block_t *block) {
    return extract_alloc(block->header);
}

static bool extract_prev_alloc(word_t word) {
    return (bool)((word & prev_alloc_mask) >> 1);
}

static bool get_prev_alloc(block_t *block) {
    return extract_prev_alloc(block->header);
}

/**
 * @brief Writes an epilogue header at the given address.
 *
 * The epilogue header has size 0, and is marked as allocated.
 *
 * @param[out] block The location to write the epilogue header
 */
static void write_epilogue(block_t *block) {
    dbg_requires(block != NULL);
    dbg_requires((char *)block == mem_heap_hi() - 7);
    block->header = pack(0, true, false);
}

/**
 * @brief Writes a block starting at the given address.
 *
 * This function writes both a header and footer, where the location of the
 * footer is computed in relation to the header.
 *
 * @param[out] block The location to begin writing the block header
 * @param[in] size The size of the new block
 * @param[in] alloc The allocation status of the new block
 * @pre The block must be a valid block, and size should be greater than zero.
 */
static void write_block(block_t *block, size_t size, bool alloc,
                        bool prev_alloc) {
    dbg_requires(block != NULL);
    dbg_requires(size > 0);
    block->header = pack(size, alloc, prev_alloc);
    if (!alloc) {
        word_t *footerp = header_to_footer(block);
        *footerp = pack(size, alloc, prev_alloc);
    }
}

/**
 * @brief Finds the next consecutive block on the heap.
 *
 * This function accesses the next block in the "implicit list" of the heap
 * by adding the size of the block.
 *
 * @param[in] block A block in the heap
 * @return The next consecutive block on the heap
 * @pre The block is not the epilogue
 */
static block_t *find_next(block_t *block) {
    dbg_requires(block != NULL);
    dbg_requires(get_size(block) != 0 &&
                 "Called find_next on the last block in the heap");
    return (block_t *)((char *)block + get_size(block));
}

/**
 * @brief Writes next block of the block starting at given address
 * @param[in] block A block in the heap
 * @param[in] alloc The allocated bit of current block
 */
static void write_next_block(block_t *block, bool alloc) {
    block_t *next_block = find_next(block);
    write_block(next_block, get_size(next_block), get_alloc(next_block), alloc);
}

/**
 * @brief Finds the footer of the previous block on the heap.
 * @param[in] block A block in the heap
 * @return The location of the previous block's footer
 */
static word_t *find_prev_footer(block_t *block) {
    // Compute previous footer position as one word before the header
    return &(block->header) - 1;
}

/**
 * @brief Finds the previous consecutive block on the heap.
 *
 * This is the previous block in the "implicit list" of the heap.
 *
 * If the function is called on the first block in the heap, NULL will be
 * returned, since the first block in the heap has no previous block!
 *
 * The position of the previous block is found by reading the previous
 * block's footer to determine its size, then calculating the start of the
 * previous block based on its size.
 *
 * @param[in] block A block in the heap
 * @return The previous consecutive block in the heap.
 */
static block_t *find_prev(block_t *block) {
    dbg_requires(block != NULL);
    word_t *footerp = find_prev_footer(block);

    // Return NULL if called on first block in the heap
    if (extract_size(*footerp) == 0) {
        return NULL;
    }

    return footer_to_header(footerp);
}

/*
 * ---------------------------------------------------------------------------
 *                        END SHORT HELPER FUNCTIONS
 * ---------------------------------------------------------------------------
 */

/******** The remaining content below are helper and debug routines ********/
void print_heap() {
    block_t *block;
    for (block = heap_start; get_size(block) > 0; block = find_next(block)) {
        printf("block address: %p, size: %zu, allocated: %zu, prev block "
               "allocated: %zu, next: %p\n",
               (void *)block, get_size(block), (size_t)get_alloc(block),
               (size_t)get_prev_alloc(block), find_next(block));
    }
}

/**
 * @brief Insert block to segregated free list.
 * @param[in] block A block in the heap
 */
static void insert_free(block_t *block) {
    if (block == NULL) {
        return;
    }

    size_t size = get_size(block);
    size_t index;
    size_t class_size;
    for (index = 0; index < seglist_max; index++) {
        class_size = min_block_size << (index + 1);
        // find the appropriate seglist
        if ((size < class_size) || (index == (seglist_max - 1))) {
            if (seglist[index] == NULL) {
                seglist[index] = block;
                block->next = NULL;
                block->prev = NULL;
                return;
            } else {
                block->next = seglist[index];
                seglist[index]->prev = block;
                block->prev = NULL;
                seglist[index] = block;
                return;
            }
        }
    }
}

/**
 * @brief Remove block from segregated free list.
 * @param[in] block A block in the heap
 */
static void remove_free(block_t *block) {
    size_t size = get_size(block);
    size_t index;
    size_t class_size;
    for (index = 0; index < seglist_max; index++) {
        class_size = min_block_size << (index + 1);
        // find the appropriate seglist
        if ((size < class_size) || (index == (seglist_max - 1))) {
            if (block == NULL || seglist[index] == NULL) {
                return;
            }
            block_t *prev = block->prev;
            block_t *next = block->next;
            if (prev == NULL && next == NULL) {
                seglist[index] = NULL;
                return;
            }
            if (prev == NULL && next != NULL) {
                seglist[index] = next;
                next->prev = NULL;
                return;
            }
            if (prev != NULL && next == NULL) {
                prev->next = NULL;
                return;
            }
            if (prev != NULL && next != NULL) {
                prev->next = next;
                next->prev = prev;
                return;
            }
        }
    }
}

/**
 * @brief Coalesce blocks if there are consecutive free blocks.
 * @param[in] block A block in the heap
 * @return Coalesced block if previous/next block is free.
 */
static block_t *coalesce_block(block_t *block) {
    bool prev_alloc = get_prev_alloc(block);
    block_t *prev_block;
    if (!prev_alloc) {
        prev_block = find_prev(block);
    }

    block_t *next_block = find_next(block);
    bool next_alloc = get_alloc(next_block);

    size_t coalesce_size = get_size(block);

    if (prev_alloc && next_alloc) {
        // printf("debug 1\n");
        write_block(block, coalesce_size, false, true);
        write_block(next_block, get_size(next_block), true, false);
    }
    if (prev_alloc && (!next_alloc)) {
        // printf("debug 2\n");
        coalesce_size += get_size(next_block);
        remove_free(next_block);
        write_block(block, coalesce_size, false, true);
    }
    if ((!prev_alloc) && next_alloc) {
        // printf("debug 3\n");
        coalesce_size += get_size(prev_block);
        remove_free(prev_block);
        write_block(prev_block, coalesce_size, false,
                    get_prev_alloc(prev_block));
        write_block(next_block, get_size(next_block), true, false);
        block = prev_block;
    }
    if ((!prev_alloc) && (!next_alloc)) {
        // printf("debug 4\n");
        coalesce_size += (get_size(prev_block) + get_size(next_block));
        remove_free(prev_block);
        remove_free(next_block);
        write_block(prev_block, coalesce_size, false,
                    get_prev_alloc(prev_block));
        block = prev_block;
    }
    return block;
}

/**
 * @brief Extend heap when no fit free block is found.
 * @param[in] size size (bytes) of extension of heap
 * @return Free block after extend heap
 */
static block_t *extend_heap(size_t size, bool prev_alloc) {
    void *bp;

    // Allocate an even number of words to maintain alignment
    size = round_up(size, dsize);
    if ((bp = mem_sbrk(size)) == (void *)-1) {
        return NULL;
    }

    // Initialize free block header/footer
    block_t *block = payload_to_header(bp);
    write_block(block, size, false, prev_alloc);

    // Create new epilogue header
    block_t *block_next = find_next(block);
    write_epilogue(block_next);

    // Coalesce in case the previous block was free
    block = coalesce_block(block);

    return block;
}

/**
 * @brief Split block if free block found is too large.
 * @param[in] block A block in the heap
 * @param[in] asize Size of block to be allocated
 * @pre The block is allocated and size of block is greater than zero.
 * @post The first block of splited blocks is allocated.
 */
static void split_block(block_t *block, size_t asize) {
    dbg_requires(get_alloc(block));
    dbg_requires(asize > 0);

    size_t block_size = get_size(block);

    if ((block_size - asize) >= min_block_size) {
        bool prev_alloc = get_prev_alloc(block);
        write_block(block, asize, true, prev_alloc);

        block_t *block_next = find_next(block);
        write_block(block_next, block_size - asize, false, true);
        insert_free(block_next);
        write_next_block(block_next, false);
    }

    dbg_ensures(get_alloc(block));
}

/**
 * @brief Find fit free block for malloc using first fit.
 * @param[in] asize Size of block to be allocated
 * @return Free block found fit for malloc
 */
static block_t *find_fit(size_t asize) {
    // first fit
    block_t *block;
    size_t index = 0;
    size_t class_size;
    while (index < seglist_max) {
        class_size = min_block_size << (index + 1);
        if ((asize < class_size) || (index == (seglist_max - 1))) {
            for (block = seglist[index]; block != NULL; block = block->next) {
                if (asize <= get_size(block)) {
                    return block;
                }
            }
        }
        index++;
    }
    return NULL; // no fit found
}

void print_list() {
    block_t *block;
    size_t index;
    for (index = 0; index < seglist_max; index++) {
        size_t min_class_size = min_block_size << index;
        size_t max_class_size = min_block_size << (index + 1);
        printf("list[%zu]: (%zu, %zu)\n", index, min_class_size,
               max_class_size);
        for (block = seglist[index]; block != NULL; block = block->next) {
            printf("block address: %p, size: %zu, allocated: %zu, prev block "
                   "allocated: %zu, next: %p, "
                   "prev: %p\n",
                   (void *)block, get_size(block), (size_t)get_alloc(block),
                   (size_t)get_prev_alloc(block), block->next, block->prev);
        }
    }
}

/**
 * @brief Scans the heap and checks it for possible errors.
 * @param[in] line Line number where this function is called
 * @return True if no error found after checking heap; otherwise, false.
 */
bool mm_checkheap(int line) {
    // check prologue and epilogue
    word_t *prologue = (word_t *)mem_heap_lo();
    word_t *epilogue = (word_t *)(mem_heap_hi() - 7);
    if ((get_size((block_t *)prologue) != 0) ||
        (!get_alloc((block_t *)prologue))) {
        printf("wrong prologue\n");
        return false;
    }
    if ((get_size((block_t *)epilogue) != 0) ||
        (!get_alloc((block_t *)epilogue))) {
        printf("wrong epilogue\n");
        return false;
    }

    block_t *block;
    size_t index;
    size_t free_blocks_in_heap = 0;
    size_t free_blocks_in_list = 0;
    for (block = heap_start; get_size(block) > 0; block = find_next(block)) {
        if ((size_t)header_to_payload(block) % dsize != 0) {
            printf("wrong address alignment\n");
            return false;
        }
        if (((void *)block < mem_heap_lo()) ||
            ((void *)block > mem_heap_hi())) {
            printf("block out of heap boundries\n");
            return false;
        }
        if (get_size(block) < min_block_size) {
            printf("block size less than min block size\n");
            return false;
        }
        if (block->header != *header_to_footer(block)) {
            printf("header and footer do not match\n");
            return false;
        }
        if (!get_alloc(block)) {
            block_t *next = find_next(block);
            if (!get_alloc(next)) {
                printf("consecutive free blocks\n");
                return false;
            }
            free_blocks_in_heap++;
        }
    }

    for (index = 0; index < seglist_max; index++) {
        size_t min_class_size = min_block_size << index;
        size_t max_class_size = min_block_size << (index + 1);
        for (block = seglist[index]; block != NULL; block = block->next) {
            // check next/previous pointers are cosistent
            block_t *next = block->next;
            if (next != NULL && next->prev != block) {
                printf("next/previous pointers not consistent\n");
                return false;
            }

            // check all free pointers within heap boundries
            if (((void *)block < mem_heap_lo()) ||
                ((void *)block > mem_heap_hi())) {
                printf("block out of heap boundries\n");
                return false;
            }

            // check blocks in each list bucket fall within bucket size range
            if ((get_size(block) < min_class_size) ||
                ((get_size(block) >= max_class_size) &&
                 (index != (seglist_max - 1)))) {
                printf("block not in the right size class\n");
                return false;
            }

            // count free blocks in free list
            if (!get_alloc(block)) {
                free_blocks_in_list++;
            }
        }
    }

    // check if free blocks in heap and free list match
    if (free_blocks_in_heap != free_blocks_in_list) {
        printf(
            "free blocks in heap and free blocks in free list do not match\n");
        printf("free blokcks in heap : %zu, in free list : %zu\n",
               free_blocks_in_heap, free_blocks_in_list);
        return false;
    }

    return true;
}

/**
 * @brief Initialize heap.
 * @return True if heap is initialized; otherwise, false.
 */
bool mm_init(void) {
    // Create the initial empty heap
    word_t *start = (word_t *)(mem_sbrk(2 * wsize));

    if (start == (void *)-1) {
        return false;
    }

    start[0] = pack(0, true, false); // Heap prologue (block footer)
    start[1] = pack(0, true, false); // Heap epilogue (block header)

    // Heap starts with first "block header", currently the epilogue
    heap_start = (block_t *)&(start[1]);

    // Extend the empty heap with a free block of chunksize bytes
    block_t *block = extend_heap(chunksize, true);
    if (block == NULL) {
        return false;
    }

    // Initialize segregated list
    for (size_t i = 0; i < seglist_max; i++) {
        seglist[i] = NULL;
    }

    insert_free(block);
    write_next_block(block, false);

    return true;
}

/**
 * @brief Allocate a block with payload of at least size bytes.
 * @param[in] size Size (bytes) of payload
 * @return A pointer to an allocated block payload of at least size bytes
 */
void *malloc(size_t size) {
    dbg_requires(mm_checkheap(__LINE__));

    size_t asize;      // Adjusted block size
    size_t extendsize; // Amount to extend heap if no fit is found
    block_t *block;
    void *bp = NULL;

    // Initialize heap if it isn't initialized
    if (heap_start == NULL) {
        mm_init();
    }

    // Ignore spurious request
    if (size == 0) {
        dbg_ensures(mm_checkheap(__LINE__));
        return bp;
    }

    // Adjust block size to include overhead and to meet alignment requirements
    asize = round_up(size + wsize, dsize);
    asize = max(asize, min_block_size);

    // Search the free list for a fit
    block = find_fit(asize);

    // If no fit is found, request more memory, and then and place the block
    if (block == NULL) {
        // Always request at least chunksize
        extendsize = max(asize, chunksize);
        bool prev_alloc = extract_prev_alloc(*((word_t *)(mem_heap_hi() - 7)));
        block = extend_heap(extendsize, prev_alloc);
        // extend_heap returns an error
        if (block == NULL) {
            return bp;
        }
    } else {
        remove_free(block);
        write_next_block(block, true);
    }

    // The block should be marked as free
    dbg_assert(!get_alloc(block));

    // Mark block as allocated
    size_t block_size = get_size(block);
    write_block(block, block_size, true, get_prev_alloc(block));
    write_next_block(block, true);

    // Try to split the block if too large
    split_block(block, asize);

    bp = header_to_payload(block);

    dbg_ensures(mm_checkheap(__LINE__));

    // print_heap();
    return bp;
}

/**
 * @brief Free the block pointed to by bp.
 * @param[in] bp A pointer to the block to be freed
 */
void free(void *bp) {
    dbg_requires(mm_checkheap(__LINE__));

    if (bp == NULL) {
        return;
    }

    block_t *block = payload_to_header(bp);
    size_t size = get_size(block);

    // The block should be marked as allocated
    dbg_assert(get_alloc(block));

    // Mark the block as free
    write_block(block, size, false, get_prev_alloc(block));
    write_next_block(block, false);

    // Try to coalesce the block with its neighbors
    block = coalesce_block(block);

    insert_free(block);
    write_next_block(block, false);

    dbg_ensures(mm_checkheap(__LINE__));
}

/**
 * @brief Free the block pointed to by ptr followed by malloc a block
 * with payload of at least size bytes.
 * @param[in] ptr A pointer to block to be freed
 * @param[in] size Size (bytes) of payload
 * @return A pointer to an allocated region of at least size bytes.
 */
void *realloc(void *ptr, size_t size) {
    block_t *block = payload_to_header(ptr);
    size_t copysize;
    void *newptr;

    // If size == 0, then free block and return NULL
    if (size == 0) {
        free(ptr);
        return NULL;
    }

    // If ptr is NULL, then equivalent to malloc
    if (ptr == NULL) {
        return malloc(size);
    }

    // Otherwise, proceed with reallocation
    newptr = malloc(size);

    // If malloc fails, the original block is left untouched
    if (newptr == NULL) {
        return NULL;
    }

    // Copy the old data
    copysize = get_payload_size(block); // gets size of old payload
    if (size < copysize) {
        copysize = size;
    }
    memcpy(newptr, ptr, copysize);

    // Free the old block
    free(ptr);

    return newptr;
}

/**
 * @brief Allocates memory for an array of elements of size bytes each.
 * @param[in] elements Number of elements of array
 * @param[in] size Size (bytes) of each element of array
 * @return A pointer to the allocated memory
 */
void *calloc(size_t elements, size_t size) {
    void *bp;
    size_t asize = elements * size;

    if (elements == 0) {
        return NULL;
    }
    if (asize / elements != size) {
        // Multiplication overflowed
        return NULL;
    }

    bp = malloc(asize);
    if (bp == NULL) {
        return NULL;
    }

    // Initialize all bits to 0
    memset(bp, 0, asize);

    return bp;
}

/*
 *****************************************************************************
 * Do not delete the following super-secret(tm) lines!                       *
 *                                                                           *
 * 53 6f 20 79 6f 75 27 72 65 20 74 72 79 69 6e 67 20 74 6f 20               *
 *                                                                           *
 * 66 69 67 75 72 65 20 6f 75 74 20 77 68 61 74 20 74 68 65 20               *
 * 68 65 78 61 64 65 63 69 6d 61 6c 20 64 69 67 69 74 73 20 64               *
 * 6f 2e 2e 2e 20 68 61 68 61 68 61 21 20 41 53 43 49 49 20 69               *
 *                                                                           *
 * 73 6e 27 74 20 74 68 65 20 72 69 67 68 74 20 65 6e 63 6f 64               *
 * 69 6e 67 21 20 4e 69 63 65 20 74 72 79 2c 20 74 68 6f 75 67               *
 * 68 21 20 2d 44 72 2e 20 45 76 69 6c 0a c5 7c fc 80 6e 57 0a               *
 *                                                                           *
 *****************************************************************************
 */
