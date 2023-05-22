package halil.todolist.domain.member.login.cookie;

import halil.todolist.domain.member.entity.Member;
import halil.todolist.domain.member.repository.MemberRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletResponse;

@Service
@RequiredArgsConstructor
public class CookieService {

    private final MemberRepository memberRepository;
    // 영속 쿠키: 만료 날짜를 입력하면 해당 날짜까지 유지
    // 세션 쿠키: 만료 날짜를 생략하면 브라우저 종료시 까지만 유지

    /*
     * 쿠키 로그인
     * Cookie 의 이름은 memberId로 생성
     */
    @Transactional
    public Member login(HttpServletResponse response, String email, String password) {
        Member member = checkMember(email, password);
        if (member == null) {
            // Exception 처리
        }

        Cookie cookie = new Cookie("memberId", String.valueOf(member.getId()));
        response.addCookie(cookie);
        return member;
    }

    private Member checkMember(String email, String password) {
        Member member = memberRepository.findByEmail(email).get();
        if (member.getPassword().equals(password)) {
            return member;
        } else {
            return null;
        }
    }

    public void logout(HttpServletResponse response) {
        expireCookie(response);
    }

    private void expireCookie(HttpServletResponse response) {
        Cookie cookie = new Cookie("memberId", null);
        cookie.setMaxAge(0);
        response.addCookie(cookie);
    }
}
