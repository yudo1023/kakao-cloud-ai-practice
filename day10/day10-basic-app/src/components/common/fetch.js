//./components/common/fetch.js

import { useFetch } from "../../hooks/useFetch";

const Fetch = () => {
    const [data, loading, error] = useFetch("https://jsonplaceholder.typicode.com/posts");

    if (loading) return <p>로딩 중...</p>;
    if (error) return <p>에러 발생: {error}</p>;

    return (
        <div>
            <h1>Posts</h1>
            <ul>
                {data.map((post) => (
                    <li key={post.id}>
                    <h3>{post.title}</h3>
                    <p>{post.body}</p>
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default Fetch;