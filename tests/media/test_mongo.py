"""Tests for `MongoMediaStore` against an in-memory `mongomock-motor` client.

`MongoMediaStore` stores blobs as manual sha256-addressed chunks over plain
async collection operations (no GridFS), so it reaches full coverage against
`mongomock-motor` without a running mongod. `_mock_client` is the single
type-boundary shim: the fake is not a `pymongo.AsyncMongoClient`, but the
store only uses the async collection surface both share.
"""

from __future__ import annotations

import pytest
from mongomock_motor import AsyncMongoMockClient
from pymongo import AsyncMongoClient

from pydantic_ai_harness.media import MediaContext, MediaStore, MongoMediaStore, media_uri_for, parse_media_uri

pytestmark = pytest.mark.anyio

_MISSING_URI = 'media+sha256://' + ('0' * 64)


def _mock_client() -> AsyncMongoClient[dict[str, object]]:
    return AsyncMongoMockClient()  # pyright: ignore[reportUnknownVariableType, reportReturnType]


class TestMongoMediaStoreConstruction:
    def test_requires_exactly_one_of_client_or_db_url(self) -> None:
        with pytest.raises(ValueError, match='exactly one'):
            MongoMediaStore(database='t')
        with pytest.raises(ValueError, match='exactly one'):
            MongoMediaStore(client=_mock_client(), db_url='mongodb://x', database='t')

    def test_requires_database(self) -> None:
        with pytest.raises(ValueError, match='`database=` is required'):
            MongoMediaStore(client=_mock_client())

    def test_rejects_invalid_collection_name(self) -> None:
        with pytest.raises(ValueError, match='invalid collection name'):
            MongoMediaStore(client=_mock_client(), database='t', collection='bad-name')

    def test_rejects_non_positive_chunk_size(self) -> None:
        with pytest.raises(ValueError, match='`chunk_size_bytes` must be positive'):
            MongoMediaStore(client=_mock_client(), database='t', chunk_size_bytes=0)

    async def test_db_url_owns_client_and_aclose_closes_it(self) -> None:
        store = MongoMediaStore(db_url='mongodb://localhost:59017', database='t')
        await store.aclose()  # owned client -> closed

    async def test_shared_client_aclose_is_noop(self) -> None:
        store = MongoMediaStore(client=_mock_client(), database='t')
        await store.aclose()  # shared client -> left open


class TestMongoMediaStoreRoundTrip:
    async def test_put_get_round_trip(self) -> None:
        store = MongoMediaStore(client=_mock_client(), database='t')
        data = b'hello mongo bytes'
        uri = await store.put(data, context=MediaContext(media_type='application/octet-stream'))
        assert uri == media_uri_for(data)
        assert await store.get(uri) == data

    async def test_multi_chunk_split_and_reassembly(self) -> None:
        """A blob larger than `chunk_size_bytes` splits into ordered chunks and rejoins."""
        client = _mock_client()
        store = MongoMediaStore(client=client, database='t', chunk_size_bytes=4)
        data = bytes(range(256)) * 40  # 10_240 bytes -> 2560 chunks of 4
        uri = await store.put(data)
        chunk_count = await client['t']['media_chunks'].count_documents({})
        assert chunk_count == (len(data) + 3) // 4
        assert await store.get(uri) == data

    async def test_dedup_is_idempotent(self) -> None:
        client = _mock_client()
        store = MongoMediaStore(client=client, database='t', chunk_size_bytes=4)
        data = b'duplicate me'
        first = await store.put(data)
        second = await store.put(data)  # hits the early-return dedup path
        assert first == second
        assert await client['t']['media'].count_documents({}) == 1
        assert await client['t']['media_chunks'].count_documents({}) == (len(data) + 3) // 4

    async def test_exists(self) -> None:
        store = MongoMediaStore(client=_mock_client(), database='t')
        uri = await store.put(b'present')
        assert await store.exists(uri) is True
        assert await store.exists(_MISSING_URI) is False

    async def test_get_missing_raises(self) -> None:
        store = MongoMediaStore(client=_mock_client(), database='t')
        with pytest.raises(FileNotFoundError):
            await store.get(_MISSING_URI)

    async def test_reassembly_mismatch_raises(self) -> None:
        """A `size_bytes` that disagrees with the chunk bytes surfaces as a loud error."""
        client = _mock_client()
        store = MongoMediaStore(client=client, database='t')
        uri = await store.put(b'twelve chars')
        digest = parse_media_uri(uri)
        await client['t']['media'].update_one({'_id': digest}, {'$set': {'size_bytes': 999}})
        with pytest.raises(ValueError, match='reassembly mismatch'):
            await store.get(uri)

    async def test_metadata_round_trips(self) -> None:
        store = MongoMediaStore(client=_mock_client(), database='t')
        uri = await store.put(
            b'tagged',
            context=MediaContext(media_type='image/png', metadata={'origin': 'user', 'tenant': 'acme'}),
        )
        assert await store.get_metadata(uri) == {'origin': 'user', 'tenant': 'acme'}

    async def test_metadata_empty_when_not_supplied(self) -> None:
        store = MongoMediaStore(client=_mock_client(), database='t')
        uri = await store.put(b'no tags')
        assert await store.get_metadata(uri) == {}

    async def test_get_metadata_missing_raises(self) -> None:
        store = MongoMediaStore(client=_mock_client(), database='t')
        with pytest.raises(FileNotFoundError):
            await store.get_metadata(_MISSING_URI)

    async def test_custom_collection_name(self) -> None:
        client = _mock_client()
        store = MongoMediaStore(client=client, database='t', collection='blobs')
        uri = await store.put(b'in a custom collection')
        assert await client['t']['blobs'].count_documents({}) == 1
        assert await client['t']['blobs_chunks'].count_documents({}) >= 1
        assert await store.get(uri) == b'in a custom collection'


class TestMongoMediaStorePublicUrl:
    async def test_without_resolver_returns_none(self) -> None:
        store = MongoMediaStore(client=_mock_client(), database='t')
        assert await store.public_url(media_uri_for(b'x')) is None

    async def test_with_resolver_uses_it(self) -> None:
        from pydantic_ai_harness.media import make_static_public_url

        store = MongoMediaStore(
            client=_mock_client(),
            database='t',
            public_url=make_static_public_url('https://cdn.example.com'),
        )
        uri = media_uri_for(b'p')
        digest = parse_media_uri(uri)
        assert await store.public_url(uri) == f'https://cdn.example.com/{digest}.bin'


class TestMongoMediaStoreProtocol:
    def test_satisfies_media_store_protocol(self) -> None:
        store: MediaStore = MongoMediaStore(client=_mock_client(), database='t')
        assert isinstance(store, MediaStore)


class TestMediaLazyExport:
    def test_mongo_media_store_lazily_exported(self) -> None:
        import pydantic_ai_harness.media as media

        assert media.MongoMediaStore is MongoMediaStore

    def test_unknown_attribute_raises(self) -> None:
        import pydantic_ai_harness.media as media

        with pytest.raises(AttributeError, match='has no attribute'):
            _ = media.NoSuchStore  # type: ignore[attr-defined]
