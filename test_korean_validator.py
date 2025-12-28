"""Test Korean language support in EmbeddingValidator."""

from embedding.validators import EmbeddingValidator
from domain import Fragment, View
import uuid


def test_validator():
    """Test both Korean and English validation patterns."""
    v = EmbeddingValidator()

    # Test cases: (content, expected_eligible, description)
    tests = [
        # Korean boilerplate
        ('저작권 © 2024 출판사', False, 'Korean copyright'),
        ('저작권 소유', False, 'Korean rights reserved'),
        ('페이지 123', False, 'Korean page number'),
        ('123 쪽', False, 'Korean page with number first'),
        ('그림 3 참조', False, 'Korean figure reference'),
        ('표 1 참고', False, 'Korean table reference'),
        ('3장 참조', False, 'Korean chapter reference'),
        ('[주석]', False, 'Korean annotation bracket'),
        ('다음과 같이', False, 'Korean as follows alone'),
        ('1.', False, 'List number alone'),

        # Valid Korean technical content
        ('이 함수는 HTTP 요청을 처리합니다', True, 'Valid Korean technical text'),
        ('리스트의 요소는 정적 배열입니다', True, 'Valid Korean technical text 2'),
        ('API 호출 시 인증 토큰이 필요합니다', True, 'Valid Korean technical text 3'),
        ('그림 3의 분포를 분석하면 패턴이 보입니다', True, 'Valid text with reference keyword'),

        # English boilerplate
        ('Copyright © 2024', False, 'English copyright'),
        ('See Figure 3', False, 'English reference'),
        ('page 42', False, 'English page number'),
        ('All rights reserved', False, 'English rights'),

        # Valid English technical content
        ('This function handles HTTP requests', True, 'Valid English technical text'),
        ('The array contains static elements', True, 'Valid English technical text 2'),
    ]

    print('Testing Korean and English patterns:')
    print('=' * 80)

    passed = 0
    failed = 0

    for content, expected_eligible, description in tests:
        frag = Fragment(
            id=str(uuid.uuid4()),
            concept_id='test-concept',
            content=content,
            view=View.TEXT,
            language=None,
            order=0
        )
        result = v.is_eligible(frag)
        status = 'PASS' if result == expected_eligible else 'FAIL'

        if result == expected_eligible:
            passed += 1
        else:
            failed += 1
            reason = v.get_ineligibility_reason(frag)

        # Use safe ASCII printing to avoid Windows console encoding issues
        safe_content = content.encode('ascii', 'replace').decode('ascii')
        print(f'[{status}] {description}')
        print(f'      Content: "{safe_content}"')
        print(f'      Eligible: {result} (expected: {expected_eligible})')
        if result != expected_eligible:
            print(f'      Reason: {reason}')
        print()

    print('=' * 80)
    print(f'Results: {passed} passed, {failed} failed out of {len(tests)} tests')
    return failed == 0


if __name__ == '__main__':
    success = test_validator()
    exit(0 if success else 1)
