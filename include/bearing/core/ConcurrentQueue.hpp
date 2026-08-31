#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <utility>

namespace bearing {

// ConcurrentQueue<T> —— 线程安全队列，支持无界和有界两种模式：
//   max_size == 0 → 无界：push() 永不因"队列满"而阻塞
//                   （内存上限由操作系统保证）；
//   max_size >  0 → 有界：容量为 N，当队列满时 push() 会阻塞等待消费者
//                   取走数据（背压），防止长时间运行时内存无限增长。
//
// 当前流水线中每条队列只有"一个生产者 + 一个消费者"（SPSC），
// 本类也可正确处理更多并发方，但未为此额外优化。
template <typename T>
class ConcurrentQueue {
public:
    // max_size == 0 → 无界；max_size > 0 → 有界，容量 N，满了产生背压。
    explicit ConcurrentQueue(std::size_t max_size = 0)
        : max_size_(max_size) {}

    // push() —— 入队一个元素。
    // 有界模式下若队列已满且未 stop，会阻塞等待消费者取走数据（背压）。
    // 返回值契约：
    //   true  = 元素已成功入队，调用方可累加"已推送"计数器；
    //   false = 队列已 stop，元素未入队，调用方不应再推送（也不应累加计数器）。
    bool push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this]() {
            return stopped_ || max_size_ == 0 || queue_.size() < max_size_;
        });

        if (stopped_) {
            return false;
        }

        queue_.push(std::move(item));
        lock.unlock();
        // 唤醒所有等待者，即使将来不是严格的"单生产者单消费者"用法也不会漏唤醒
        // （当前管线是 SPSC，notify_one 也正确，但 notify_all 更稳健、代价可忽略）。
        condition_.notify_all();
        return true;
    }

    // pop() —— 取出队头元素。
    // 队列空且未 stop 时阻塞；仅当已 stop 且队列取空后返回 false（让消费者
    // drain 完剩余数据再退出，不会丢失 stop 前已经入队的元素）。
    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this]() {
            return stopped_ || !queue_.empty();
        });

        if (queue_.empty()) {
            return false;
        }

        item = std::move(queue_.front());
        queue_.pop();
        lock.unlock();
        // 同 push()：notify_all 保证即使有多个生产者等待也不会漏唤醒。
        condition_.notify_all();
        return true;
    }

    // stop() —— 置 stopped_ 并唤醒所有等待者，让阻塞中的 push()/pop()
    // 尽快返回（push 返回 false；pop 在 drain 完剩余元素后返回 false）。
    // 此操作幂等：多次调用安全。
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        condition_.notify_all();
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::queue<T> queue_;
    std::size_t max_size_ = 0;
    bool stopped_ = false;
};

}  // namespace bearing
