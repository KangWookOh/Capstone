package halil.todolist.domain.member.login.session;

import halil.todolist.domain.member.entity.Member;
import halil.todolist.domain.member.exception.session.LoginUserNotFound;
import halil.todolist.domain.member.repository.MemberRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

@Service
@RequiredArgsConstructor
public class SessionService {

    private final String SESSION_COOKIE_NAME = "SessionId";
    private final PasswordEncoder passwordEncoder;
    private Map<String, Object> sessionStore = new ConcurrentHashMap<>();


    private final MemberRepository memberRepository;

    @Transactional
    public Member login(String email, String password, HttpServletResponse response) {
        Member member = checkMember(email, password);
        if (member == null) {
            throw new LoginUserNotFound();
        }

        createSession(member, response);
        return member;
    }

    /**
     * 세선 생성
     * @param value : Member 정보
     * @param response
     */
    @Transactional
    public void createSession(Object value, HttpServletResponse response) {
        Member member = (Member) value;
        // 세션 생성, 저장
        String sessionId = UUID.randomUUID().toString();
        sessionStore.put(sessionId, member);

        // 쿠키 생성
        Cookie cookie = new Cookie(SESSION_COOKIE_NAME, sessionId);
        response.addCookie(cookie);
    }

    /**
     * 세션 조회
     * @param request
     * @return : Member 의 정보 반환
     */
    public Object getSession(HttpServletRequest request) {
        Cookie sessionCookie = findCookie(request, SESSION_COOKIE_NAME);
        if (sessionCookie == null) {
            return null;
        }
        // sessionCookie ==> (name = Cookie Name, value = UUID)
        return sessionStore.get(sessionCookie.getValue());
    }

    /**
     * 로그아웃(세션 만료)
     * @param request
     */
    public void logout(HttpServletRequest request) {
        // HttpSession : session.invalidate();  --> 세션을 무효화 할 수 있지 사용자가 요청을 또 보내면 정보가 그대로 남아있다.
        request.getSession().invalidate();      // 세션을 무효화 하고
        request.getSession(true);        // 새로운 세션ID 발급
    }

    /**
     * 세션 만료
     * @param request
     */
    public void expire(HttpServletRequest request) {
        Cookie cookie = findCookie(request, SESSION_COOKIE_NAME);
        if (cookie != null) {
            sessionStore.remove(cookie.getValue());
        }
    }

    private Cookie findCookie(HttpServletRequest request, String cookieName) {
        if (request.getCookies() == null) {
            return null;
        }

        return Arrays.stream(request.getCookies())
                .filter(cookie -> cookie.getName().equals(cookieName))
                .findAny()
                .orElse(null);
    }

    private Member checkMember(String email, String password) {
        // null 일경우 Exception 처리
        memberRepository.findByEmail(email).orElseThrow(() -> new LoginUserNotFound());

        Member member = memberRepository.findByEmail(email).get();
        // member.getPassword().equals()
        if (passwordEncoder.matches(password, member.getPassword())) {
            return member;
        } else {
            return null;
        }
    }
}
